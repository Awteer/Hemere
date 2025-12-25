import numpy as np
import joblib
import os
import sys
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.aero.tr797_aerodynamics_solver import TR797AerodynamicsSolver
    from src.aero.bending_moment_calculator import BendingMomentCalculator
except ImportError as e:
    print(f"Critical Error: Required modules not found.\nDetail: {e}")
    sys.exit(1)

MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# =========================================================
# 【必須】Pickle復元用クラス定義 (学習時と完全一致させる)
# =========================================================
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        log_ei = X[:, 0]
        r = X[:, 1]
        log_r = np.log10(r + 1e-9)
        thickness_index = log_ei - 3 * log_r
        return np.column_stack((X, thickness_index))

# =========================================================
# Beta 最適化クラス
# =========================================================
class BetaOptimizer:
    def __init__(self, velocity_ms, span_m, fixed_weight_kg):
        self.V = velocity_ms
        self.b = span_m
        self.W_fixed_other = fixed_weight_kg
        self.rho = 1.17
        self.g = 9.80665
        
        # 構造パラメータ
        self.epsilon_limit = 0.003
        self.CD0_fixed = 0.01
        self.W_fixed_wing = 8.0
        
        # ソルバー初期化
        self.num_aero_points = 50
        self.aero_solver = TR797AerodynamicsSolver(span_m, num_points=self.num_aero_points) 
        self.bmd_calc = BendingMomentCalculator(span_m, fixed_wing_weight_kg=self.W_fixed_wing)
        
        # EOSモデル読み込み
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: EOS Model not found at {MODEL_PATH}")
            self.eos_model = None
        else:
            try:
                # クラス定義があるので正常に読み込める
                self.eos_model = joblib.load(MODEL_PATH)
                print(f"EOS Model Loaded Successfully.")
            except Exception as e:
                print(f"Error Loading EOS Model: {e}")
                self.eos_model = None

    def get_geometry(self, y_coords):
        b_half = self.b / 2.0
        eta = np.abs(y_coords) / b_half
        chord = 0.90 * (1 - eta) + 0.30 * eta
        diameter_mm = np.clip((chord * 0.11) * 1000.0, 30.0, 130.0)
        return chord, diameter_mm

    def predict_spar_weight_eos(self, required_EI_N_mm2, diameters_mm):
        """
        EOSモデルを使用して重量分布を予測する。
        【修正】モデルはSI単位(Nmm^2)で学習されたため、単位変換は不要。
        """
        if self.eos_model is None:
            return 0.5 * np.zeros_like(diameters_mm)
        
        # 1. 入力データの準備 [Log10(EI_N), Diameter]
        # そのままSI単位でLogをとる
        safe_EI = np.clip(required_EI_N_mm2, 1e6, None)
        X_basic = np.column_stack((np.log10(safe_EI), diameters_mm))
        
        # 2. 予測 (Pipeline内で PhysicsFeatureEngineer が自動適用される)
        pred_log_w = self.eos_model.predict(X_basic)
        
        # 3. 単位変換 (Log10(kg/m) -> kg/m)
        pred_w = 10**pred_log_w
        
        return pred_w

    def calculate_structural_weight(self, lift_dist_N_m, y_coords, W_spar_guess_kg_m):
        chords, diameters_mm = self.get_geometry(y_coords)
        
        # BMD計算 (N単位)
        _, M_dist_Nm, _ = self.bmd_calc.calculate_moment(
            y_aero_m=y_coords, 
            lift_dist_N_m=lift_dist_N_m, 
            spar_weight_dist_kg_m=W_spar_guess_kg_m 
        )
        
        # 必要EI算出 (SI単位: Nmm^2)
        M_Nmm = np.abs(M_dist_Nm) * 1000.0
        r_outer_mm = diameters_mm / 2.0
        EI_req_N_mm2 = M_Nmm * r_outer_mm / self.epsilon_limit
        
        # 重量予測 (SI単位のままモデルへ)
        W_spar_new_kg_m = self.predict_spar_weight_eos(EI_req_N_mm2, diameters_mm)
        
        # 積分
        from scipy.integrate import simpson
        W_spar_total_kg = simpson(W_spar_new_kg_m, x=y_coords) * 2.0
        
        return W_spar_total_kg, W_spar_new_kg_m, M_dist_Nm, EI_req_N_mm2

    def evaluate_beta(self, beta, verbose=False):
        # 初期推定値
        W_guess_kg = self.W_fixed_other + self.W_fixed_wing + 15.0 
        y_points = self.aero_solver.y
        W_spar_guess_kg_m = np.full(self.num_aero_points, 0.5)
        
        final_Di = 0.0
        final_W_spar = 0.0
        
        # 連成ループ
        for i in range(20):
            L_total_req_N = W_guess_kg * self.g
            
            # 空力
            gamma_phys, Di, e = self.aero_solver.solve_gamma_and_drag(
                beta, L_total_req_N, self.rho, self.V
            )
            if np.isnan(Di): return 1e9, None
            
            lift_dist_N_m = self.rho * self.V * gamma_phys
            
            # 構造
            W_spar_total_new, W_spar_dist_new, M_dist, EI_req = self.calculate_structural_weight(
                lift_dist_N_m, y_points, W_spar_guess_kg_m
            )
            
            W_total_new_kg = self.W_fixed_other + self.W_fixed_wing + W_spar_total_new
            
            # 収束判定
            if abs(W_total_new_kg - W_guess_kg) < 0.05:
                W_guess_kg = W_total_new_kg
                final_Di = Di
                final_W_spar = W_spar_total_new
                W_spar_guess_kg_m = W_spar_dist_new
                break
            
            # 緩和
            W_guess_kg = 0.5 * W_total_new_kg + 0.5 * W_guess_kg
            W_spar_guess_kg_m = 0.5 * W_spar_dist_new + 0.5 * W_spar_guess_kg_m

        # 出力作成
        chords, _ = self.get_geometry(y_points)
        from scipy.integrate import simpson
        S_ref = simpson(chords, x=y_points) * 2.0 
        
        q = 0.5 * self.rho * self.V**2
        Dp = q * S_ref * self.CD0_fixed
        Power = (final_Di + Dp) * self.V
        
        details = {
            "Power": Power,
            "TotalWeight": W_guess_kg,
            "SparWeight": final_W_spar,
            "IndDrag": final_Di,
            "ParasiteDrag": Dp,
            "TotalDrag": final_Di + Dp,
            "LiftDist": lift_dist_N_m,
            "SparWeightDist": W_spar_guess_kg_m,
            "MomentDist": M_dist,
            "EI_Req": EI_req, # Nmm^2
            "y_coords": y_points
        }
        return Power, details

    def optimize(self):
        # (以前と同じため省略、evaluate_betaを呼び出す)
        pass 
        # ※実際のファイルでは元のoptimizeメソッドを含めてください
        
if __name__ == "__main__":
    # デバッグ実行
    optimizer = BetaOptimizer(velocity_ms=7.5, span_m=34.0, fixed_weight_kg=80.0)
    optimizer.evaluate_beta(1.0, verbose=True)