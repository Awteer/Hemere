import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# モデルパス
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# =========================================================
# 【必須】学習時と同じ特徴量生成関数を定義
# =========================================================
def add_physics_features(X):
    """
    物理的なヒントを追加する関数 (学習時と全く同じロジック)
    """
    log_ei = X[:, 0]
    r = X[:, 1]
    
    # log(EI) - 3*log(R)
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

# =========================================================
# ソルバークラス
# =========================================================
class EosSolver:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        print(f"Loading Eos Optimizer Engine ({MODEL_PATH})...")
        self.model = joblib.load(MODEL_PATH)
        
        # 探索範囲 (mm)
        self.r_min = 30.0
        self.r_max = 130.0
        self.r_step = 0.5 # 0.5mm刻みで探索

    def find_optimal_diameter(self, target_EI_kgf_mm2):
        """
        指定された剛性(EI)に対して、最も軽くなる直径(R)を見つける
        """
        # 1. 探索用グリッド作成
        r_scan = np.arange(self.r_min, self.r_max + self.r_step, self.r_step)
        
        # 2. 入力データの作成 [Log10(EI), R]
        # EIは固定、Rだけ変化させる
        log_ei = np.log10(target_EI_kgf_mm2)
        ei_col = np.full_like(r_scan, log_ei)
        
        X_scan = np.column_stack((ei_col, r_scan))
        
        # 3. Eosモデルで爆速推論
        predicted_weights = self.model.predict(X_scan)
        
        # 4. 最適解の探索 (Weightが最小になるインデックス)
        best_idx = np.argmin(predicted_weights)
        
        best_r = r_scan[best_idx]
        min_weight = predicted_weights[best_idx]
        
        result = {
            "Target_EI": target_EI_kgf_mm2,
            "Optimal_R": best_r,
            "Estimated_Weight": min_weight,
            "Scan_R": r_scan,         # グラフ描画用
            "Scan_W": predicted_weights # グラフ描画用
        }
        
        return result

    def plot_optimization_curve(self, result):
        """
        最適化のプロセス（直径 vs 重量カーブ）を可視化する
        """
        r = result["Scan_R"]
        w = result["Scan_W"]
        best_r = result["Optimal_R"]
        best_w = result["Estimated_Weight"]
        target_ei = result["Target_EI"]

        plt.figure(figsize=(10, 6))
        plt.plot(r, w, label=f'Eos Prediction (EI={target_ei:.1e})', linewidth=2)
        
        # 最適点に赤丸
        plt.scatter(best_r, best_w, color='red', s=100, zorder=5, label=f'Optimal: R={best_r}mm')
        plt.axvline(best_r, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f"Single Section Optimization\nTarget Stiffness EI = {target_ei:.2e} [kgf mm^2]")
        plt.xlabel("Diameter R [mm]")
        plt.ylabel("Estimated Weight [kg/m]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# =========================================================
# メイン実行部
# =========================================================
if __name__ == "__main__":
    solver = EosSolver()

    # --- テストケース: 要求剛性を入力 ---
    target_EI = float(input("テスト剛性>>"))
    
    print("\n" + "="*50)
    print(f" Optimization Request: EI = {target_EI:.2e}")
    print("="*50)
    
    # 最適化実行
    res = solver.find_optimal_diameter(target_EI)
    
    print(f"✅ Optimal Diameter : {res['Optimal_R']:.1f} mm")
    print(f"✅ Estimated Weight : {res['Estimated_Weight']:.4f} kg/m")
    
    # グラフ表示
    solver.plot_optimization_curve(res)
