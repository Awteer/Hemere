import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import itertools

# ==========================================
# 1. 物理モデル (SparCalculator)
# ==========================================
class SparCalculator:
    def __init__(self):
        self.mat_props = np.array([13000, 1900, 900, 22000, 1900, 800])
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])

    def calculate_spec(self, ply_counts, R):
        n_layers = 11
        thickness = ply_counts * 0.111
        thickness[0] = 0.125
        thickness[10] = 0.125
        
        inner = np.zeros(n_layers)
        outer = np.zeros(n_layers)
        current_inner = R # input R is radius or diameter? 
        # previous code treated R as "inner radius" or similar logic.
        # Assuming R is "radius" based on logic.
        
        inner[0] = current_inner
        outer[0] = inner[0] + 2 * thickness[0]
        
        for i in range(1, n_layers):
            inner[i] = inner[i-1] + 2 * thickness[i-1]
            outer[i] = inner[i] + 2 * thickness[i]

        Ix = (np.pi / 64) * (outer**4 - inner**4) * (1 - np.cos(np.deg2rad(self.pp)))
        
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
        
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights[i] = 0.001496 * 1.0 * rS[i]
            else:
                weights[i] = 0.001559 * 1.0 * rS[i]
        
        total_weight = np.sum(weights) / 1000.0
        return total_EI, total_weight

# ==========================================
# 2. データ生成
# ==========================================
def generate_dataset():
    calc = SparCalculator()
    data_EI = []
    data_Weight = []
    data_R = []
    
    # 桁径のバリエーション (半径 mm)
    # 40mm(直径80) ～ 70mm(直径140) 程度を想定
    radii = [30, 40, 45, 50, 55, 60] 
    radii = radii * 2
    
    print("Generating 2D dataset (EI, R)...")
    
    base_plies = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # 層3～7 を0～5枚で変動 (広めに探索)
    iter_ranges = [range(0, 6) for _ in range(5)]
    
    # データ生成ループ
    for R in radii:
        for p_counts in itertools.product(*iter_ranges):
            current_plies = base_plies.copy()
            for k, val in enumerate(p_counts):
                current_plies[k+2] = val + 1
            
            EI, W = calc.calculate_spec(current_plies, R)
            data_EI.append(EI)
            data_Weight.append(W)
            data_R.append(R) # 半径を記録

    return np.array(data_EI), np.array(data_R), np.array(data_Weight)

# ==========================================
# 3. 2変数モデルの定義と解析
# ==========================================

# モデル式: W = A * EI^B * R^C + D
def model_func_2d(X, a, b, c, d):
    ei, r = X
    # スケーリングして計算安定化 (EIは10^10オーダーなので)
    # ここでは生の値でfitさせるが、初期値(p0)を工夫する
    return a * (ei ** b) * (r ** c) + d

def analyze_2d_model(top_percent=10.0):
    EI_raw, R_raw, W_raw = generate_dataset()
    
    # --- 1. 全体フィット ---
    # 初期値推測: B=1, C=-2 (物理則 W ∝ EI / R^2)
    p0 = [1e-5, 1.0, -2.0, 0.0] 
    
    try:
        popt_all, _ = curve_fit(model_func_2d, (EI_raw, R_raw), W_raw, p0=p0, maxfev=20000)
    except Exception as e:
        print(f"Initial fitting failed: {e}")
        return

    # --- 2. エリート選抜 (Top X%) ---
    # 予測値よりどれだけ軽いか（残差）で判定
    W_pred_all = model_func_2d((EI_raw, R_raw), *popt_all)
    residuals = W_raw - W_pred_all
    
    cutoff_index = int(len(residuals) * (top_percent / 100.0))
    sorted_indices = np.argsort(residuals)
    best_indices = sorted_indices[:cutoff_index]
    
    EI_best = EI_raw[best_indices]
    R_best = R_raw[best_indices]
    W_best = W_raw[best_indices]
    
    print(f"Selected {len(EI_best)} elite designs.")

    # --- 3. エリートフィット (Refitting) ---
    try:
        popt_best, _ = curve_fit(model_func_2d, (EI_best, R_best), W_best, p0=popt_all, maxfev=20000)
    except Exception as e:
        print(f"Refitting failed: {e}")
        return
    
    W_pred_best = model_func_2d((EI_best, R_best), *popt_best)
    
    # 誤差評価
    residuals_best = W_best - W_pred_best
    rel_error = (residuals_best / W_best) * 100
    
    # 結果表示
    print("-" * 40)
    print(f"Model: Weight = A * EI^B * R^C + D")
    print(f"Optimized Parameters (Elite {top_percent}%):")
    print(f"  A = {popt_best[0]:.4e}")
    print(f"  B = {popt_best[1]:.4f} (Expected ~1.0)")
    print(f"  C = {popt_best[2]:.4f} (Expected ~-2.0)")
    print(f"  D = {popt_best[3]:.4f}")
    
    rmse = np.sqrt(np.mean(residuals_best**2))
    print(f"RMSE: {rmse:.6f} kg/m")
    
    # --- プロット作成 ---
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: 予測 vs 実測 (対角線に近いほど良い)
    ax1 = fig.add_subplot(131)
    ax1.scatter(W_best, W_pred_best, alpha=0.5, s=5, c='blue')
    # y=x line
    lims = [min(W_best.min(), W_pred_best.min()), max(W_best.max(), W_pred_best.max())]
    ax1.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax1.set_xlabel('Actual Weight [kg/m]')
    ax1.set_ylabel('Predicted Weight [kg/m]')
    ax1.set_title('Actual vs Predicted (Elite)')
    ax1.grid(True)
    
    # Plot 2: 相対誤差のヒートマップ (EI vs R)
    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(EI_best, R_best, c=rel_error, cmap='coolwarm', s=10, vmin=-5, vmax=5, alpha=0.8)
    ax2.set_xlabel('Bending Stiffness EI')
    ax2.set_ylabel('Radius R [mm]')
    ax2.set_title('Relative Error Map [%]')
    plt.colorbar(sc, ax=ax2, label='Error %')
    ax2.grid(True)
    
    # Plot 3: 相対誤差分布 (ヒストグラム)
    ax3 = fig.add_subplot(133)
    ax3.hist(rel_error, bins=50, color='green', alpha=0.7)
    ax3.set_xlabel('Relative Error [%]')
    ax3.set_title(f'Error Distribution (Std: {np.std(rel_error):.2f}%)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_2d_model(top_percent=10.0)