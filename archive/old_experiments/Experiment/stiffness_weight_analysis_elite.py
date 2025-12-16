import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools

# ==========================================
# 1. 物理モデル (SparCalculator)
#    ※前回のコードと同じロジック
# ==========================================

class SparCalculator:
    def __init__(self):
        # Material: [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        # EI.mのMaterial配列: [13000 1900 900 22000 1900 800]
        self.mat_props = np.array([13000, 1900, 900, 22000, 1900, 800])
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])

    def calculate_spec(self, ply_counts, R):
        n_layers = 11
        thickness = ply_counts * 0.111
        thickness[0] = 0.125
        thickness[10] = 0.125
        
        inner = np.zeros(n_layers)
        outer = np.zeros(n_layers)
        current_inner = R
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
    
    # バリエーションを少し広げて、より多くのサンプルを作る
    radii = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130] 
    
    print("Generating extensive dataset...")
    
    base_plies = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # 層3～7 (index 2~6) を0～4枚で変動
    iter_ranges = [range(0, 5) for _ in range(5)]
    
    for R in radii:
        for p_counts in itertools.product(*iter_ranges):
            current_plies = base_plies.copy()
            for k, val in enumerate(p_counts):
                current_plies[k+2] = val + 1
            
            EI, W = calc.calculate_spec(current_plies, R)
            data_EI.append(EI)
            data_Weight.append(W)

    return np.array(data_EI), np.array(data_Weight)

# ==========================================
# 3. エリート選抜と解析 (Elite Selection)
# ==========================================

def fit_function(x, a, b, c):
    return a * (x ** b) + c

def analyze_best_designs(top_percent=10.0):
    # 1. 全データの生成
    EI_raw, W_raw = generate_dataset()
    
    # 2. 一次フィッティング（全体傾向の把握）
    p0 = [0.0001, 0.5, 0.1]
    try:
        popt_all, _ = curve_fit(fit_function, EI_raw, W_raw, p0=p0, maxfev=10000)
    except:
        print("Initial fitting failed.")
        return

    # 3. 残差（Residuals）の計算
    # 実際の重さ - 予測された重さ
    # マイナスであればあるほど「予測より軽い＝優秀」
    W_pred_all = fit_function(EI_raw, *popt_all)
    residuals = W_raw - W_pred_all
    
    # 4. エリートデータの抽出
    # 残差が小さい（負に大きい）順にソートして、上位X%を取得
    cutoff_index = int(len(residuals) * (top_percent / 100.0))
    sorted_indices = np.argsort(residuals) # 小さい順（優秀な順）
    best_indices = sorted_indices[:cutoff_index]
    
    EI_best = EI_raw[best_indices]
    W_best = W_raw[best_indices]
    
    # 5. エリートデータに対する再フィッティング (Refitting)
    try:
        popt_best, _ = curve_fit(fit_function, EI_best, W_best, p0=popt_all, maxfev=10000)
    except:
        print("Refitting failed.")
        return
        
    W_pred_best = fit_function(EI_best, *popt_best)
    
    # --- 誤差評価 ---
    residuals_best = W_best - W_pred_best
    rmse = np.sqrt(np.mean(residuals_best**2))
    max_error = np.max(np.abs(residuals_best))
    
    # 決定係数 R^2 (for best subset)
    ss_res = np.sum(residuals_best**2)
    ss_tot = np.sum((W_best - np.mean(W_best))**2)
    r_squared_best = 1 - (ss_res / ss_tot)
    
    print(f"--- Analysis Result (Top {top_percent}%) ---")
    print(f"Selected {len(EI_best)} designs out of {len(EI_raw)}")
    print(f"Optimized Parameters: A={popt_best[0]:.4e}, B={popt_best[1]:.4f}, C={popt_best[2]:.4f}")
    print(f"R-squared (Elite): {r_squared_best:.5f}")
    print(f"RMSE: {rmse:.6f} kg/m")
    print(f"Max Error: {max_error:.6f} kg/m")

    # --- プロット作成 ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: 全体 vs エリート
    axs[0].scatter(EI_raw, W_raw, alpha=0.1, s=2, color='gray', label='All Designs')
    axs[0].scatter(EI_best, W_best, alpha=0.6, s=3, color='blue', label=f'Top {top_percent}% Designs')
    
    # ソートしてラインを描画
    sort_idx_all = np.argsort(EI_raw)
    sort_idx_best = np.argsort(EI_best)
    
    # 全体フィット（点線）
    axs[0].plot(EI_raw[sort_idx_all], W_pred_all[sort_idx_all], 
                color='black', linestyle='--', linewidth=1, label='General Fit')
    # エリートフィット（赤実線）
    axs[0].plot(EI_best[sort_idx_best], W_pred_best[sort_idx_best], 
                color='red', linewidth=2, label='Elite Fit (Surrogate)')
    
    axs[0].set_xlabel('Bending Stiffness EI [N mm^2]')
    axs[0].set_ylabel('Linear Density [kg/m]')
    axs[0].set_title('Elite Selection & Surrogate Modeling')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: エリートフィットの誤差分布
    # 相対誤差(%)
    rel_error = (residuals_best / W_best) * 100
    axs[1].scatter(EI_best, rel_error, alpha=0.4, s=5, color='green')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_xlabel('Bending Stiffness EI')
    axs[1].set_ylabel('Relative Error [%]')
    axs[1].set_title(f'Prediction Error for Top {top_percent}% Designs')
    axs[1].set_ylim(-5, 5) # ±5%にズーム
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_best_designs(top_percent=10.0)