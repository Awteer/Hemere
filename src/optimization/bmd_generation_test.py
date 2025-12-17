import numpy as np
import joblib
import os
import sys
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 開発済みクラスをインポート
try:
    from src.aero.tr797_aerodynamics_solver import TR797AerodynamicsSolver
    from src.aero.bending_moment_calculator import BendingMomentCalculator
except ImportError as e:
    print(f"Critical Error: 必要なモジュールが見つかりません。\n詳細: {e}")
    sys.exit(1)

# EOSモデルパス
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# =========================================================
# EOSモデル用 ヘルパー関数 (Pickle読み込みに必須)
# =========================================================
def add_physics_features(X):
    """
    モデル学習時と同じ特徴量生成ロジック。
    ※重要: パイプライン内で自動で呼ばれる場合があるため、
    予測時に手動で呼ぶと二重適用になる可能性があります。
    ここでは joblib.load のために定義だけ残します。
    """
    log_ei = X[:, 0]
    r_diameter = X[:, 1]
    
    # 学習時の特徴量エンジニアリングと一致させる
    log_r = np.log10(r_diameter + 1e-9)
    
    # 厚み指数 ( EI / R^3 ~ E*t )
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    return 10**x

# =========================================================
# Beta 最適化クラス
# =========================================================
class BetaOptimizer:
    def __init__(self, velocity_ms, span_m, fixed_weight_kg):
        # 基本諸元
        self.V = velocity_ms
        self.b = span_m
        self.W_fixed_other = fixed_weight_kg # パイロット + フェアリング・尾翼・駆動系
        self.rho = 1.17                      # 空気密度 [kg/m^3]
        self.g = 9.80665
        
        # 構造・設計パラメータ
        self.epsilon_limit = 0.003           # 設計許容歪み (3000µε)
        self.CD0_fixed = 0.01                # 有害抗力係数 (仮)
        self.W_fixed_wing = 8.0              # 翼の固定重量（リブ、スキン、フィルム等）[kg]
        
        # 空力ソルバー
        self.num_aero_points = 50
        self.aero_solver = TR797AerodynamicsSolver(span_m, num_points=self.num_aero_points) 
        
        # BMD計算機
        self.bmd_calc = BendingMomentCalculator(span_m, fixed_wing_weight_kg=self.W_fixed_wing)
        
        # 構造重量モデル (EOS)
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: EOS Model not found at {MODEL_PATH}")
            self.eos_model = None
        else:
            try:
                self.eos_model = joblib.load(MODEL_PATH)
                print(f"EOS Model Loaded Successfully: {MODEL_PATH}")
            except Exception as e:
                print(f"Error Loading EOS Model: {e}")
                self.eos_model = None

    def get_geometry(self, y_coords):
        """
        スパン位置yにおける幾何形状 (Chord, Diameter) を返す。
        """
        b_half = self.b / 2.0
        eta = np.abs(y_coords) / b_half
        
        # テーパー翼定義
        c_root = 0.90
        c_tip = 0.30
        chord = c_root * (1 - eta) + c_tip * eta
        
        # 桁直径 (Diameter) の決定
        # 弦長の 11% 程度を確保
        diameter_mm_raw = (chord * 0.11) * 1000.0
        
        # EOSの学習範囲 (30mm ~ 130mm) にクリップ
        diameter_mm = np.clip(diameter_mm_raw, 30.0, 130.0)
        
        return chord, diameter_mm

    def predict_spar_weight_eos(self, required_EI_N_mm2, diameters_mm):
        """
        EOSモデルを使用して重量分布を予測する。
        修正: 入力EI(Nmm^2)をモデル学習時の単位(kgfmm^2)に変換する。
        """
        if self.eos_model is None:
            return 0.5 * np.zeros_like(diameters_mm)
        
        # --- 単位変換 (重要) ---
        # モデルは kgf・mm^2 で学習されているため、N・mm^2 を 9.80665 で割る
        required_EI_kgf_mm2 = required_EI_N_mm2 / self.g
        
        # 1. 入力データの準備 [Log10(EI_kgf), Diameter]
        # EIが小さすぎるとLogでエラーになるので下限クリップ
        # (kgf換算で 1e6 ~ 1e7 程度が下限目安)
        safe_EI_kgf = np.clip(required_EI_kgf_mm2, 1e6, None)
        
        X_basic = np.column_stack((np.log10(safe_EI_kgf), diameters_mm))
        
        # 2. XGBoost予測 (Log10(kg/m)が返ってくる)
        pred_log_w = self.eos_model.predict(X_basic)
        
        # 3. 単位変換 (Log10(kg/m) -> kg/m)
        pred_w = 10**pred_log_w
        
        return pred_w

    def calculate_structural_weight(self, lift_dist_N_m, y_coords, W_spar_guess_kg_m):
        """
        揚力分布と推定重量分布からBMDを計算し、必要EIを算出し、
        EOSモデルを使って新しい重量分布を予測する。
        """
        # 1. 形状取得
        chords, diameters_mm = self.get_geometry(y_coords)
        
        # 2. BMD計算 (Net Load = Lift - Weight_Total)
        _, M_dist_Nm, _ = self.bmd_calc.calculate_moment(
            y_aero_m=y_coords, 
            lift_dist_N_m=lift_dist_N_m, 
            spar_weight_dist_kg_m=W_spar_guess_kg_m 
        )
        
        # 3. 必要EIの算出 (Strain Design)
        M_Nmm = np.abs(M_dist_Nm) * 1000.0 # 絶対値
        r_outer_mm = diameters_mm / 2.0
        
        # EI = M * y / epsilon
        EI_req = M_Nmm * r_outer_mm / self.epsilon_limit
        
        # 4. EOSモデルで重量予測 [kg/m]
        W_spar_new_kg_m = self.predict_spar_weight_eos(EI_req, diameters_mm)
        
        # 5. 総重量積分 (Simpson則)
        from scipy.integrate import simpson
        W_spar_total_kg = simpson(W_spar_new_kg_m, x=y_coords) * 2.0
        
        return W_spar_total_kg, W_spar_new_kg_m, M_dist_Nm, EI_req

    def evaluate_beta(self, beta, verbose=False):
        """
        あるbeta値における全機性能と総パワーを評価する。
        """
        # --- 1. 初期推定 ---
        W_guess_kg = self.W_fixed_other + self.W_fixed_wing + 15.0 
        y_points = self.aero_solver.y
        W_spar_guess_kg_m = np.full(self.num_aero_points, 0.5) # 初期値: 0.5kg/m
        
        final_Di = 0.0
        final_W_spar = 0.0
        is_converged = False
        
        # --- 2. 重量収束連成ループ ---
        for i in range(20):
            L_total_req_N = W_guess_kg * self.g
            
            # 2-1. 空力計算
            gamma_phys, Di, e = self.aero_solver.solve_gamma_and_drag(
                beta, L_total_req_N, self.rho, self.V
            )
            
            if np.isnan(Di): return 1e9, None # エラーガード
            
            lift_dist_N_m = self.rho * self.V * gamma_phys
            
            # 2-2. 構造重量計算
            W_spar_total_new, W_spar_dist_new, M_dist, EI_req = self.calculate_structural_weight(
                lift_dist_N_m, 
                y_points, 
                W_spar_guess_kg_m
            )
            
            W_total_new_kg = self.W_fixed_other + self.W_fixed_wing + W_spar_total_new
            
            # 2-3. 収束判定
            diff = abs(W_total_new_kg - W_guess_kg)
            if diff < 0.05:
                W_guess_kg = W_total_new_kg
                final_Di = Di
                final_W_spar = W_spar_total_new
                W_spar_guess_kg_m = W_spar_dist_new
                is_converged = True
                break
            
            # 2-4. 緩和更新
            W_guess_kg = 0.5 * W_total_new_kg + 0.5 * W_guess_kg
            W_spar_guess_kg_m = 0.5 * W_spar_dist_new + 0.5 * W_spar_guess_kg_m

        # --- 3. 目的関数: 総パワー ---
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
            "EI_Req": EI_req,
            "y_coords": y_points,
            "SpanEfficiency": e,
            "WingArea": S_ref
        }
        
        return Power, details

    def optimize(self):
        print(f"--- Beta Optimization Start ---")
        
        def objective(b):
            p, _ = self.evaluate_beta(b, verbose=False)
            return p
        
        # 探索範囲拡大: 0.5 ~ 3.0 (1.0に張り付くのを防ぐため)
        res = minimize_scalar(objective, bounds=(0.8, 1.1), method='bounded', options={'xatol': 1e-3})
        
        print("\n" + "="*50)
        print(f"Optimization Finished! Optimal Beta : {res.x:.4f}")
        print("="*50)
        
        P_min, d = self.evaluate_beta(res.x, verbose=True)
        
        # --- 結果表示 ---
        print(f" Performance Metrics:")
        print(f"  Min Power      : {P_min:.2f} W")
        print(f"  Total Drag     : {d['TotalDrag']:.2f} N")
        print(f"    - Induced    : {d['IndDrag']:.2f} N")
        print(f"    - Parasite   : {d['ParasiteDrag']:.2f} N")
        print(f"  Lift/Drag (L/D): {(d['TotalWeight']*self.g)/d['TotalDrag']:.1f}")
        print("-" * 30)
        print(f" Weight Breakdown:")
        print(f"  Total Weight   : {d['TotalWeight']:.2f} kg")
        print(f"  Spar Weight    : {d['SparWeight']:.2f} kg (Avg: {d['SparWeight']/self.b:.3f} kg/m)")
        print(f"  Fixed Weight   : {self.W_fixed_other + self.W_fixed_wing:.2f} kg")
        print("-" * 30)
        print(f" Structural Check:")
        max_moment = np.max(np.abs(d['MomentDist']))
        print(f"  Max Bending Moment: {max_moment:.1f} Nm")
        print(f"  Max Spar Weight   : {np.max(d['SparWeightDist']):.3f} kg/m")
        print("="*50)
        
        self.plot_results(d, res.x)
        return res.x, d

    def plot_results(self, d, best_beta):
        y = d['y_coords']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Optimization Result (Beta={best_beta:.2f}, Power={d['Power']:.1f}W)", fontsize=16)
        
        # 1. 揚力分布
        ax = axes[0, 0]
        ax.plot(y, d['LiftDist'], 'g-', label='Lift (L\')')
        ax.set_title("Lift Distribution")
        ax.set_xlabel("Span [m]")
        ax.set_ylabel("Lift [N/m]")
        ax.grid(True)
        
        # 2. BMD
        ax = axes[0, 1]
        ax.plot(y, d['MomentDist'], 'b-', label='Bending Moment')
        ax.set_title("Bending Moment Distribution")
        ax.set_xlabel("Span [m]")
        ax.set_ylabel("Moment [Nm]")
        ax.grid(True)
        
        # 3. 必要EI
        ax = axes[1, 0]
        ax.semilogy(y, d['EI_Req'], 'm-', label='Required EI')
        ax.set_title("Required Stiffness (Log Scale)")
        ax.set_xlabel("Span [m]")
        ax.set_ylabel("EI [Nmm^2]")
        ax.grid(True)

        # 4. スパー重量分布
        ax = axes[1, 1]
        ax.plot(y, d['SparWeightDist'], 'r-', label='Spar Weight')
        ax.set_title(f"Spar Weight (Avg: {d['SparWeight']/self.b:.3f} kg/m)")
        ax.set_xlabel("Span [m]")
        ax.set_ylabel("Weight [kg/m]")
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import os

    # 必要ならパスを追加 (環境に合わせて調整してください)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # 先ほど作成したクラスをインポート (同じファイルにある場合はそのまま使用)
    # from your_script_name import BetaOptimizer

    def run_debug_checks():
        print("========================================================")
        print("   BetaOptimizer Debugging Suite")
        print("========================================================")

    # 1. インスタンス生成
    # テスト用に少し軽めの設定で初期化
    try:
        optimizer = BetaOptimizer(velocity_ms=7.5, span_m=34.0, fixed_weight_kg=80.0)
        print("[OK] Instance created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to create instance: {e}")

    # ---------------------------------------------------------
    # Test 1: Geometry Check (幾何形状の確認)
    # ---------------------------------------------------------
    print("\n--- Test 1: Geometry Logic Check ---")
    try:
        y_test = np.linspace(0, 17.0, 50) # 0 to b/2
        chords, diams = optimizer.get_geometry(y_test)
        
        # チェックポイント
        print(f"  Root Chord: {chords[0]:.3f} m (Expected ~0.9)")
        print(f"  Tip Chord : {chords[-1]:.3f} m (Expected ~0.3)")
        print(f"  Min Diam  : {np.min(diams):.2f} mm (Should be >= 30.0)")
        print(f"  Max Diam  : {np.max(diams):.2f} mm (Should be <= 130.0)")
        
        if np.min(diams) < 30.0 or np.max(diams) > 130.0:
            print("  [WARNING] Diameter is out of bounds or clipping failed!")
        else:
            print("  [PASS] Geometry bounds look correct.")

    except Exception as e:
        print(f"  [ERROR] Geometry check failed: {e}")

    # ---------------------------------------------------------
    # Test 2: EOS Model Prediction Check (AIモデルの確認)
    # ---------------------------------------------------------
    print("\n--- Test 2: EOS Model Prediction Check ---")
    if optimizer.eos_model is None:
        print("  [SKIP] EOS Model not loaded. Skipping this test.")
    else:
        try:
            # テストデータ: EI=1e9 (強), EI=1e7 (弱), 直径=80mm
            test_EI = np.array([1.0e9, 1.0e7])
            test_D  = np.array([80.0,  80.0])
            
            weights = optimizer.predict_spar_weight_eos(test_EI, test_D)
            
            print(f"  Input EI: {test_EI}")
            print(f"  Output Weight [kg/m]: {weights}")
            
            # 論理チェック: 剛性が高い方が重いはず
            if weights[0] > weights[1]:
                print("  [PASS] Heavier load results in heavier spar.")
            else:
                print("  [FAIL] Physics violation: Higher stiffness resulted in lighter spar.")
                
        except Exception as e:
            print(f"  [ERROR] Prediction failed: {e}")

    # ---------------------------------------------------------
    # Test 3: Structural Step Check (構造計算単体の確認)
    # ---------------------------------------------------------
    print("\n--- Test 3: Single Structural Calculation Step ---")
    try:
        # ダミーの揚力分布作成 (楕円分布に近いもの)
        # L ~ 100kgf -> 1000N
        y_pts = optimizer.aero_solver.y
        # 簡易楕円: sqrt(1 - (y/b)^2)
        lift_dummy = 60.0 * np.sqrt(1 - (y_pts / (34/2))**2) 
        w_spar_dummy = np.full_like(y_pts, 0.5) # 仮のスパー重量
        
        W_total, W_dist_new, M_dist, EI_req = optimizer.calculate_structural_weight(
            lift_dummy, y_pts, w_spar_dummy
        )
        
        print(f"  Total Spar Weight: {W_total:.3f} kg")
        print(f"  Max Bending Moment: {np.max(np.abs(M_dist)):.2f} Nm")
        print(f"  Max Required EI: {np.max(EI_req):.2e} Nmm^2")
        
        # 境界条件チェック: 翼端のモーメントはほぼ0のはず
        if abs(M_dist[-1]) < 1.0: 
            print("  [PASS] Tip moment is approximately zero.")
        else:
            print(f"  [FAIL] Tip moment is non-zero: {M_dist[-1]}")
            
    except Exception as e:
        print(f"  [ERROR] Structural step failed: {e}")

    # ---------------------------------------------------------
    # Test 4: Convergence Loop Check (収束ループの確認)
    # ---------------------------------------------------------
    print("\n--- Test 4: Convergence Loop (Evaluate Beta) ---")
    try:
        # ベータ=0 (楕円分布) でテスト
        beta_test = 1.0
        print(f"  Testing with beta = {beta_test}...")
        
        # verbose=True にして内部の動きを見たいが、
        # ここでは返り値が正常かを見る
        power, details = optimizer.evaluate_beta(beta_test, verbose=True)
        
        print(f"  [RESULT] Power: {power:.2f} W")
        print(f"  [RESULT] Final Total Weight: {details['TotalWeight']:.2f} kg")
        print(f"  [RESULT] Induced Drag: {details['IndDrag']:.2f} N")
        
        if np.isnan(power):
            print("  [FAIL] Optimization returned NaN.")
        else:
            print("  [PASS] Loop finished with valid numbers.")
            
    except Exception as e:
        print(f"  [ERROR] Convergence loop failed: {e}")

    print("\n========================================================")
    print("   Debugging Finished")
    print("========================================================")