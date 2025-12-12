import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

# ==========================================
# 1. 物理モデル (SparCalculator)
#    ※Rを「直径」として扱います
# ==========================================
class SparCalculator:
    def __init__(self):
        # Material: [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        # EI.mのMaterial配列: [13000 1900 900 22000 1900 800]
        self.mat_props = np.array([13000, 1900, 900, 22000, 1900, 800])
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])

    def calculate_spec(self, ply_counts, R_diameter):
        n_layers = 11
        thickness = ply_counts * 0.111
        thickness[0] = 0.125
        thickness[10] = 0.125
        
        inner = np.zeros(n_layers)
        outer = np.zeros(n_layers)
        
        # R_diameter is the Inner Diameter of the 1st layer (Mandrel Diameter)
        inner[0] = R_diameter
        outer[0] = inner[0] + 2 * thickness[0] # Diameter + 2*thickness
        
        for i in range(1, n_layers):
            inner[i] = inner[i-1] + 2 * thickness[i-1]
            outer[i] = inner[i] + 2 * thickness[i]

        # 断面二次モーメント Ix (直径D, dを使う公式: pi/64 * (D^4 - d^4))
        Ix = (np.pi / 64) * (outer**4 - inner**4) * (1 - np.cos(np.deg2rad(self.pp)))
        
        # 弾性率の適用
        E_vec = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                E_vec[i] = self.mat_props[2]
            elif idx == 2:
                E_vec[i] = self.mat_props[4]
            else:
                E_vec[i] = self.mat_props[3]
        
        EIx = Ix * E_vec
        total_EI = np.sum(EIx)
        
        # 重量計算 (断面積 A = pi/4 * (D^2 - d^2))
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights[i] = 0.001496 * 1.0 * rS[i]
            else:
                weights[i] = 0.001559 * 1.0 * rS[i]
        
        total_weight = np.sum(weights) / 1000.0 # kg/m
        return total_EI, total_weight

# ==========================================
# 2. 大規模データ生成 (Corrected Range)
# ==========================================
def generate_large_dataset(n_samples=1000000):
    calc = SparCalculator()
    
    data_EI = []
    data_R = [] # ここには直径が入ります
    data_Weight = []
    
    print(f"Generating {n_samples} samples with Corrected Range...")
    print("  - Diameter R: 40mm ~ 120mm") # 修正箇所
    print("  - Plies:      0 ~ 25 plies per layer")
    
    for _ in range(n_samples):
        # 1. 直径Rをランダムに決定 (40mm ~ 120mm)
        R_dia = random.uniform(40.0, 120.0)
        
        # 2. 積層数をランダムに決定
        # ベース: 層1,2,9,10,11は1枚固定 (例)
        current_plies = np.array([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        
        # メイン構造層 (層3～8) を0～25枚でランダムに変動
        for k in range(2, 8):
            current_plies[k] = random.randint(0, 25)
            
        EI, W = calc.calculate_spec(current_plies, R_dia)
        
        data_EI.append(EI)
        data_R.append(R_dia)
        data_Weight.append(W)

    return np.array(data_EI), np.array(data_R), np.array(data_Weight)

# ==========================================
# 3. 2変数モデルの解析 (Analysis)
# ==========================================

# モデル式: W = A * EI^B * R^C + D
# Rは直径としてフィッティングされます
def model_func_2d(X, a, b, c, d):
    ei, r = X
    return a * (ei ** b) * (r ** c) + d

def analyze_large_scale_model(top_percent=5.0):
    EI_raw, R_raw, W_raw = generate_large_dataset(n_samples=50000)
    
    # --- 1. 初期パラメータ推定 ---
    # 理論的には W ∝ EI / D^2 なので、B=1, C=-2 付近になるはず
    p0 = [1e-5, 1.0, -2.0, 0.0] 
    
    try:
        popt_all, _ = curve_fit(model_func_2d, (EI_raw, R_raw), W_raw, p0=p0, maxfev=50000)
    except Exception as e:
        print(f"Initial fitting failed: {e}")
        return

    # --- 2. エリート選抜 (Top X%) ---
    W_pred_all = model_func_2d((EI_raw, R_raw), *popt_all)
    residuals = W_raw - W_pred_all
    
    cutoff_index = int(len(residuals) * (top_percent / 100.0))
    sorted_indices = np.argsort(residuals)
    best_indices = sorted_indices[:cutoff_index]
    
    EI_best = EI_raw[best_indices]
    R_best = R_raw[best_indices]
    W_best = W_raw[best_indices]
    
    print(f"Selected {len(EI_best)} elite designs (Top {top_percent}%).")

    # --- 3. エリートフィット (Refitting) ---
    try:
        popt_best, _ = curve_fit(model_func_2d, (EI_best, R_best), W_best, p0=popt_all, maxfev=50000)
    except Exception as e:
        print(f"Refitting failed: {e}")
        return
    
    W_pred_best = model_func_2d((EI_best, R_best), *popt_best)
    
    # 誤差評価
    residuals_best = W_best - W_pred_best
    rel_error = (residuals_best / W_best) * 100
    rmse = np.sqrt(np.mean(residuals_best**2))
    
    # --- 結果出力 ---
    print("-" * 40)
    print(f"Dataset Range (Check):")
    print(f"  EI: {EI_best.min():.2e} ~ {EI_best.max():.2e}")
    print(f"  Diameter R : {R_best.min():.1f} ~ {R_best.max():.1f}")
    print(f"  Weight : {W_best.min():.4f} ~ {W_best.max():.4f}")
    print("-" * 40)
    print(f"Optimized Parameters (Elite {top_percent}%):")
    print(f"  A = {popt_best[0]:.6e}")
    print(f"  B = {popt_best[1]:.6f} (Expected ~1.0)")
    print(f"  C = {popt_best[2]:.6f} (Expected ~-2.0)")
    print(f"  D = {popt_best[3]:.6f}")
    print(f"RMSE: {rmse:.6f} kg/m")
    
    # --- プロット作成 ---
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: 予測 vs 実測
    ax1 = fig.add_subplot(131)
    ax1.scatter(W_best, W_pred_best, alpha=0.1, s=2, c='blue')
    lims = [min(W_best.min(), W_pred_best.min()), max(W_best.max(), W_pred_best.max())]
    ax1.plot(lims, lims, 'r-', alpha=0.8, zorder=10)
    ax1.set_xlabel('Actual Weight [kg/m]')
    ax1.set_ylabel('Predicted Weight [kg/m]')
    ax1.set_title(f'Actual vs Predicted (N={len(EI_best)})')
    ax1.grid(True)
    
    # Plot 2: 相対誤差マップ (EI vs Diameter)
    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(EI_best, R_best, c=rel_error, cmap='coolwarm', s=5, vmin=-5, vmax=5, alpha=0.5)
    ax2.set_xlabel('Bending Stiffness EI')
    ax2.set_ylabel('Diameter R [mm]')
    ax2.set_title('Relative Error Map [%]')
    plt.colorbar(sc, ax=ax2, label='Error %')
    ax2.grid(True)
    
    # Plot 3: 誤差ヒストグラム
    ax3 = fig.add_subplot(133)
    ax3.hist(rel_error, bins=100, color='green', alpha=0.7)
    ax3.set_xlabel('Relative Error [%]')
    ax3.set_title(f'Error Distribution (Std: {np.std(rel_error):.2f}%)')
    ax3.set_xlim(-10, 10)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_large_scale_model(top_percent=10.0)