import numpy as np
import joblib
import os
import sys
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson, cumulative_trapezoid
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
# Pickle復元用クラス
# =========================================================
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame): X = X.values
        if X.ndim == 1: X = X.reshape(1, -1)
        log_ei = X[:, 0]
        r = X[:, 1]
        log_r = np.log10(r + 1e-9)
        thickness_index = log_ei - 3 * log_r
        return np.column_stack((X, thickness_index))

# =========================================================
# Beta 最適化クラス (5分割ストレートパイプモデル)
# =========================================================
class BetaOptimizer:
    def __init__(self, velocity_ms, span_m, fixed_weight_kg, deflection_limit_m=None, n_sections=5):
        self.V = velocity_ms
        self.b = span_m
        self.W_fixed_other = fixed_weight_kg
        self.rho = 1.175
        self.g = 9.80665
        
        # --- 設計制約 ---
        self.epsilon_limit = 0.003
        
        if deflection_limit_m is None:
            self.deflection_limit_m = (span_m / 2.0) * 0.15
        else:
            self.deflection_limit_m = deflection_limit_m

        # 分割数 (5分割)
        self.n_sections = n_sections

        self.CD0_fixed = 0.01
        self.W_fixed_wing = 8.0
        
        # ソルバー (点数を分割数で割り切れるように調整: 50点)
        self.num_aero_points = 50 
        self.aero_solver = TR797AerodynamicsSolver(span_m, num_points=self.num_aero_points) 
        self.bmd_calc = BendingMomentCalculator(span_m, fixed_wing_weight_kg=self.W_fixed_wing)
        
        # AIモデル
        if not os.path.exists(MODEL_PATH):
            print(f"[Warning] EOS Model not found at {MODEL_PATH}")
            self.eos_model = None
        else:
            try:
                self.eos_model = joblib.load(MODEL_PATH)
                print(f"[Info] EOS Model Loaded Successfully.")
            except Exception as e:
                print(f"[Error] Failed to load EOS Model: {e}")
                self.eos_model = None

    def get_chord_at_y(self, y):
        """任意位置yの弦長を返す"""
        b_half = self.b / 2.0
        eta = np.abs(y) / b_half
        c_root = 0.90
        c_tip = 0.30
        return c_root * (1 - eta) + c_tip * eta

    def predict_spar_weight_eos(self, required_EI_N_mm2, diameters_mm):
        if self.eos_model is None:
            return 0.5 * np.zeros_like(diameters_mm)
        
        # 入力データの準備 (単一の値でも配列でも動くように)
        req_EI = np.atleast_1d(required_EI_N_mm2)
        diams = np.atleast_1d(diameters_mm)
        
        safe_EI = np.clip(req_EI, 1e6, None)
        log_ei = np.log10(safe_EI)
        X_basic = np.column_stack((log_ei, diams))
        
        pred_log_w = self.eos_model.predict(X_basic)
        return 10**pred_log_w

    def calculate_deflection(self, y_coords, M_dist_Nm, EI_dist_Nmm2):
        EI_Nm2 = EI_dist_Nmm2 * 1e-6
        safe_EI = np.maximum(EI_Nm2, 1.0)
        curvature = M_dist_Nm / safe_EI
        slope = cumulative_trapezoid(curvature, y_coords, initial=0)
        deflection = cumulative_trapezoid(slope, y_coords, initial=0)
        return deflection

    def calculate_structural_weight(self, lift_dist_N_m, y_coords, W_spar_guess_kg_m):
        """
        5分割ストレートパイプモデルによる重量計算
        """
        # 1. BMD計算 (初期推定重量を使用)
        _, M_dist_Nm, _ = self.bmd_calc.calculate_moment(
            y_aero_m=y_coords, 
            lift_dist_N_m=lift_dist_N_m, 
            spar_weight_dist_kg_m=W_spar_guess_kg_m 
        )
        M_Nmm = np.abs(M_dist_Nm) * 1000.0

        # --- 5分割ロジック開始 ---
        N = len(y_coords)
        # 各セクションの境界インデックス (例: 0, 10, 20, 30, 40, 50)
        indices = np.linspace(0, N, self.n_sections + 1, dtype=int)
        
        # 全領域のプロファイル配列を作成
        diameter_profile = np.zeros(N)
        EI_profile = np.zeros(N)
        weight_profile = np.zeros(N)
        
        # 各セクションごとに設計
        for i in range(self.n_sections):
            start = indices[i]
            end = indices[i+1] # endはスライス用なので+1されている
            
            # このセクション内の座標範囲
            y_section = y_coords[start:end]
            
            # --- A. 直径の決定 (一定) ---
            # セクション内で「最も翼が薄い場所」＝「最も外側 (yが大きい方)」
            # 外側にはみ出さないように、セクション外端の弦長で直径を決める
            y_outer = y_section[-1] 
            chord_min = self.get_chord_at_y(y_outer)
            
            # 直径はセクション内一定
            D_section_mm = np.clip((chord_min * 0.11) * 1000.0, 30.0, 130.0)
            
            # --- B. 必要剛性の決定 (一定) ---
            # セクション内で「最もモーメントが大きい場所」＝「最も内側 (yが小さい方)」
            # ここで壊れないように設計すれば、セクション全体で安全
            # (厳密には区間内の最大モーメントをとる)
            M_max_section = np.max(M_Nmm[start:end])
            
            r_outer = D_section_mm / 2.0
            EI_req_section = M_max_section * r_outer / (self.epsilon_limit + 1e-9)
            
            # --- C. 重量の予測 (一定) ---
            # AIモデルは配列を受け取るので、1点だけ渡して予測
            w_section = self.predict_spar_weight_eos(EI_req_section, D_section_mm)[0]
            
            # 配列に書き込み (セクション内はフラット)
            diameter_profile[start:end] = D_section_mm
            EI_profile[start:end] = EI_req_section
            weight_profile[start:end] = w_section

        # 4. たわみチェック & スケーリング
        # 階段状になった EI_profile を使ってたわみを計算
        deflection_dist = self.calculate_deflection(y_coords, M_dist_Nm, EI_profile)
        tip_deflection = deflection_dist[-1]
        
        stiffness_factor = 1.0
        constraint_mode = "Strain (5-Section)"
        
        if tip_deflection > self.deflection_limit_m:
            # 制限オーバーなら全体を嵩上げ (プロファイル形状は維持)
            stiffness_factor = tip_deflection / self.deflection_limit_m
            constraint_mode = "Deflection (5-Section)"
        
        # 最終的なプロファイル
        EI_final = EI_profile * stiffness_factor
        
        # 剛性が上がった分、重量も再計算 (直径は変わらないので、EIだけ増やす)
        # ※ここもセクションごとに再計算が必要
        weight_final = np.zeros(N)
        for i in range(self.n_sections):
            start = indices[i]
            end = indices[i+1]
            # このセクションの新しいEI
            ei_new = EI_final[start] 
            d_const = diameter_profile[start]
            
            w_new = self.predict_spar_weight_eos(ei_new, d_const)[0]
            weight_final[start:end] = w_new
            
        # 積分
        W_spar_total_kg = simpson(weight_final, x=y_coords) * 2.0
        
        return {
            "TotalWeight": W_spar_total_kg,
            "SparDist": weight_final,
            "Moment": M_dist_Nm,
            "EI_Req": EI_final,
            "EI_Strain_Raw": EI_profile, # スケーリング前の強度基準EI
            "DiameterDist": diameter_profile,
            "Deflection": deflection_dist / stiffness_factor,
            "TipDeflect": tip_deflection / stiffness_factor,
            "ConstraintMode": constraint_mode
        }

    def evaluate_beta(self, beta, verbose=False):
        W_guess_kg = self.W_fixed_other + self.W_fixed_wing + 20.0 
        y_points = self.aero_solver.y
        W_spar_guess_kg_m = np.full(self.num_aero_points, 0.6)
        
        final_data = {}
        
        for i in range(25):
            L_total_req_N = W_guess_kg * self.g
            
            gamma_phys, Di, e = self.aero_solver.solve_gamma_and_drag(
                beta, L_total_req_N, self.rho, self.V
            )
            if np.isnan(Di) or Di < 0: return 1e9, None
            
            lift_dist_N_m = self.rho * self.V * gamma_phys
            
            res = self.calculate_structural_weight(
                lift_dist_N_m, y_points, W_spar_guess_kg_m
            )
            
            W_total_new_kg = self.W_fixed_other + self.W_fixed_wing + res["TotalWeight"]
            
            if abs(W_total_new_kg - W_guess_kg) < 0.05:
                W_guess_kg = W_total_new_kg
                final_data = res
                final_data["IndDrag"] = Di
                final_data["LiftDist"] = lift_dist_N_m
                final_data["SpanEff"] = e
                break
            
            W_guess_kg = 0.5 * W_total_new_kg + 0.5 * W_guess_kg
            W_spar_guess_kg_m = 0.5 * res["SparDist"] + 0.5 * W_spar_guess_kg_m

        if not final_data: return 1e9, None

        chords = np.array([self.get_chord_at_y(y) for y in y_points])
        S_ref = simpson(chords, x=y_points) * 2.0 
        q = 0.5 * self.rho * self.V**2
        Dp = q * S_ref * self.CD0_fixed
        Power = (final_data["IndDrag"] + Dp) * self.V
        
        details = {
            "Power": Power,
            "TotalWeight": W_guess_kg,
            "SparWeight": final_data["TotalWeight"],
            "IndDrag": final_data["IndDrag"],
            "ParasiteDrag": Dp,
            "TotalDrag": final_data["IndDrag"] + Dp,
            "LiftDist": final_data["LiftDist"],
            "SparWeightDist": final_data["SparDist"],
            "MomentDist": final_data["Moment"],
            "EI_Req": final_data["EI_Req"],
            "DiameterDist": final_data["DiameterDist"],
            "DeflectionDist": final_data["Deflection"],
            "TipDeflect": final_data["TipDeflect"],
            "ConstraintMode": final_data["ConstraintMode"],
            "y_coords": y_points,
            "SpanEfficiency": final_data["SpanEff"]
        }
        return Power, details

    def optimize(self):
        print(f"--- Beta Optimization (5-Section Constant Spar) ---")
        print(f" Deflection Limit: {self.deflection_limit_m:.2f} m")
        
        def objective(b):
            p, _ = self.evaluate_beta(b)
            return p
        
        res = minimize_scalar(objective, bounds=(0.75, 1.05), method='bounded', options={'xatol': 1e-3})
        
        print("\n" + "="*60)
        print(f" Optimal Beta : {res.x:.4f}")
        print("="*60)
        
        P_min, d = self.evaluate_beta(res.x, verbose=True)
        
        print(f" [Constraint]")
        print(f"  Mode           : {d['ConstraintMode']}")
        print(f"  Tip Deflection : {d['TipDeflect']:.3f} m")
        print("-" * 30)
        print(f" [Performance]")
        print(f"  Min Power      : {P_min:.2f} W")
        print(f"  Lift/Drag      : {(d['TotalWeight']*self.g)/d['TotalDrag']:.1f}")
        print("-" * 30)
        print(f" [Weight]")
        print(f"  Total Weight   : {d['TotalWeight']:.2f} kg")
        print(f"  Spar Weight    : {d['SparWeight']:.2f} kg")
        print("="*60)
        
        self.plot_results(d, res.x)

    def plot_results(self, d, best_beta):
        y = d['y_coords']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Optimization Result (Beta={best_beta:.3f})\n{d['ConstraintMode']}", fontsize=14)
        
        # Deflection & Diameter
        ax = axes[0,0]
        ax.plot(y, d['DeflectionDist'], 'c-', lw=2, label='Deflection [m]')
        ax.set_ylabel("Deflection [m]")
        ax.axhline(self.deflection_limit_m, color='r', linestyle=':', label='Limit')
        
        ax2 = ax.twinx()
        ax2.plot(y, d['DiameterDist'], 'k--', lw=1.5, label='Diameter [mm]')
        ax2.set_ylabel("Diameter [mm]")
        ax.set_title("Deflection & Diameter (Stepped)")
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        ax.grid(True)
        
        # Moment
        ax = axes[0,1]
        ax.plot(y, d['MomentDist'], 'b-', lw=2)
        ax.set_title("Bending Moment")
        ax.set_ylabel("Moment [Nm]")
        ax.grid(True)
        
        # EI (Stepped)
        ax = axes[1,0]
        ax.semilogy(y, d['EI_Req'], 'm-', lw=2, label='EI (Stepped)')
        ax.set_title("Stiffness EI")
        ax.set_ylabel("EI [Nmm^2]")
        ax.grid(True)

        # Weight (Stepped)
        ax = axes[1,1]
        ax.plot(y, d['SparWeightDist'], 'r-', lw=2)
        ax.set_title(f"Spar Weight (Total: {d['SparWeight']:.1f}kg)")
        ax.set_ylabel("Weight [kg/m]")
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    # 5分割モデルで実行
    optimizer = BetaOptimizer(
        velocity_ms=7.5, 
        span_m=34.0, 
        fixed_weight_kg=80.0, 
        deflection_limit_m=2.5,
        n_sections=5
    )
    optimizer.optimize()