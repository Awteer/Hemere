import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import itertools
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 物理モデル (SparCalculator)
# ==========================================
class SparCalculator:
    def __init__(self):
        # Material Props in kgf/mm^2 [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        self.mat_props_kgf = np.array([13000, 1900, 900, 22000, 1900, 800])
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])
        self.g = 9.80665

    def calculate_spec(self, ply_counts, R_diameter):
        n_layers = 11
        # 厚み計算 (1層あたり0.111mm, 最内・最外は0.125mmと仮定)
        thickness = ply_counts * 0.111
        thickness[0] = 0.125
        thickness[10] = 0.125
        
        inner = np.zeros(n_layers)
        outer = np.zeros(n_layers)
        inner[0] = R_diameter
        outer[0] = inner[0] + 2 * thickness[0]
        for i in range(1, n_layers):
            inner[i] = inner[i-1] + 2 * thickness[i-1]
            outer[i] = inner[i] + 2 * thickness[i]

        # EI calculation (N*mm^2)
        Ix = (np.pi / 64) * (outer**4 - inner**4) * (1 - np.cos(np.deg2rad(self.pp)))
        
        E_vec_kgf = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                E_vec_kgf[i] = self.mat_props_kgf[2]
            elif idx == 2:
                E_vec_kgf[i] = self.mat_props_kgf[4]
            else:
                E_vec_kgf[i] = self.mat_props_kgf[3]
        
        EIx_kgf = Ix * E_vec_kgf
        total_EI_Nmm2 = np.sum(EIx_kgf) * self.g

        # Weight calculation (kg/m)
        # Density factors based on area
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights_kg_m = np.zeros(n_layers)
        
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights_kg_m[i] = 0.001496 * rS[i]
            else:
                weights_kg_m[i] = 0.001559 * rS[i]
        
        total_weight_kg_m = np.sum(weights_kg_m)
        
        return total_EI_Nmm2, total_weight_kg_m

# ==========================================
# 2. パレート最適データ生成 (カタログスペック生成)
# ==========================================
def generate_pareto_dataset():
    calc = SparCalculator()
    data = []

    # --- パラメータ探索範囲 ---
    # 直径: 30mm ~ 130mm, 0.5mm刻み
    diameters = np.arange(30.0, 130.0, 0.5)
    
    # 積層構成の総当たり
    # 全周積層 (Layer 2): 0 ~ 5枚
    ply_2_opts = range(0, 6) 
    # 集中積層 (Layer 3~8): 合計 0 ~ 2枚程度を想定
    # ここでは簡易化のため、特定の層に厚みを付加するパターンを網羅
    ply_conc_opts = range(0, 3) 

    print("Generating spar catalog (Brute Force method)...")
    
    for R in diameters:
        for p2 in ply_2_opts:
            for p_conc in ply_conc_opts:
                # ベース積層 [Glass, Carbon45, Carbon0...]
                # index: 0=Glass, 1=C45, 2=C90(全周), 3~8=C0(集中), 9,10=Glass/etc
                current_plies = np.array([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]) 
                
                # 全周積層の適用
                current_plies[2] = p2 
                
                # 集中積層の適用 (上下フランジへの追加を想定し、効果が高い層に配置)
                # 例: Layer 3 と Layer 4 に追加
                if p_conc > 0:
                     current_plies[5] = p_conc
                     current_plies[8] = p_conc
                
                EI, W = calc.calculate_spec(current_plies, R)
                data.append((EI, R, W))

    data_arr = np.array(data)
    
    # --- パレート解（下側包絡線）の抽出 ---
    # EIを細かい区間(ビン)に分け、各区間で「最小重量」のデータだけを残す
    EI_raw = data_arr[:, 0]
    W_raw = data_arr[:, 2]
    
    # 対数スケールでビンを切ると、剛性が低い領域も高い領域もバランスよく取れる
    n_bins = 2000 
    bins = np.logspace(np.log10(EI_raw.min()), np.log10(EI_raw.max()), n_bins)
    
    indices_to_keep = []
    
    print("Filtering for Pareto frontier (lightest designs only)...")
    for i in range(len(bins)-1):
        # ビンに含まれるデータを探す
        mask = (EI_raw >= bins[i]) & (EI_raw < bins[i+1])
        if np.any(mask):
            # そのビンの中で最も軽いデータのインデックスを取得
            subset_indices = np.where(mask)[0]
            # 最小重量のインデックス
            min_weight_idx = subset_indices[np.argmin(W_raw[subset_indices])]
            indices_to_keep.append(min_weight_idx)
            
    EI_best = data_arr[indices_to_keep, 0]
    R_best = data_arr[indices_to_keep, 1]
    W_best = data_arr[indices_to_keep, 2]
    
    print(f"Total combinations: {len(data)} -> Pareto optimized dataset: {len(EI_best)}")
    
    return EI_best, R_best, W_best

# ==========================================
# 3. 回帰モデル定義
# ==========================================
def model_func_2d(X, a, b, c, d,e):
    ei, r = X
    return a * (ei ** b) + e* (r ** c) + d

# ==========================================
# 4. 批判的検証クラス (Visualization)
# ==========================================
class CriticalVerifier:
    def __init__(self, EI, R, W_actual, W_pred):
        self.EI = EI
        self.R = R
        self.W_actual = W_actual
        self.W_pred = W_pred
        self.rel_error = (self.W_pred - self.W_actual) / self.W_actual * 100.0
        
    def plot_dashboard(self):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Critical Verification: Pareto Optimized Model ($W = A \cdot EI^B \cdot R^C + D$)", fontsize=16)

        # Plot 1: Accuracy
        ax1 = fig.add_subplot(2, 3, 1)
        sc1 = ax1.scatter(self.W_actual, self.W_pred, c=np.abs(self.rel_error), cmap='coolwarm', s=5, alpha=0.8, vmin=0, vmax=3)
        ax1.plot([self.W_actual.min(), self.W_actual.max()], [self.W_actual.min(), self.W_actual.max()], 'k--', lw=1)
        ax1.set_xlabel('Actual Weight (kg/m)')
        ax1.set_ylabel('Predicted Weight (kg/m)')
        ax1.set_title('Accuracy Check')
        plt.colorbar(sc1, ax=ax1, label='Abs Error %')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error Distribution
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(self.rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title(f'Error Distribution\nMean: {np.mean(self.rel_error):.2f}%, Std: {np.std(self.rel_error):.2f}%')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Bias vs Diameter
        ax3 = fig.add_subplot(2, 3, 4)
        sc3 = ax3.scatter(self.R, self.rel_error, c=self.EI, cmap='viridis', s=5, alpha=0.6)
        ax3.axhline(0, color='red', linestyle='--', lw=1)
        ax3.set_xlabel('Diameter R (mm)')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_title('Bias vs Diameter')
        plt.colorbar(sc3, ax=ax3, label='EI')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Bias vs Stiffness
        ax4 = fig.add_subplot(2, 3, 5)
        sc4 = ax4.scatter(self.EI, self.rel_error, c=self.R, cmap='plasma', s=5, alpha=0.6)
        ax4.axhline(0, color='red', linestyle='--', lw=1)
        ax4.set_xscale('log')
        ax4.set_xlabel('Stiffness EI (Nmm2) [Log]')
        ax4.set_title('Bias vs Stiffness')
        plt.colorbar(sc4, ax=ax4, label='R')
        ax4.grid(True, alpha=0.3)

        # # Plot 5: 3D Map
        # ax5 = fig.add_subplot(2, 3, (3, 6), projection='3d')
        # # Subsample for visibility
        # idx_sub = np.random.choice(len(self.EI), min(3000, len(self.EI)), replace=False)
        # p = ax5.scatter(np.log10(self.EI[idx_sub]), self.R[idx_sub], self.W_actual[idx_sub], 
        #                 c=self.rel_error[idx_sub], cmap='bwr', s=10, vmin=-3, vmax=3)
        # ax5.set_xlabel('Log10(EI)')
        # ax5.set_ylabel('Diameter R')
        # ax5.set_zlabel('Weight')
        # ax5.set_title('3D Error Map')
        # fig.colorbar(p, ax=ax5, label='Error %', shrink=0.6)

        plt.tight_layout()
        plt.show()

# ==========================================
# 5. メイン実行処理
# ==========================================
def analyze_pareto_model():
    # 1. パレート最適データの生成
    EI_best, R_best, W_best = generate_pareto_dataset()
    
    # 2. カーブフィッティング
    p0 = [1e-5, 0.8, -1.3, 0.0,0.0]
    try:
        popt_best, pcov = curve_fit(model_func_2d, (EI_best, R_best), W_best, p0=p0, maxfev=100000)
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        return

    W_pred_best = model_func_2d((EI_best, R_best), *popt_best)

    # 3. 結果出力
    print("-" * 50)
    print(f"Optimization Results (Pareto Frontier):")
    print(f"Dataset Range:")
    print(f"  EI: {EI_best.min():.2e} ~ {EI_best.max():.2e}")
    print(f"  R : {R_best.min():.1f} ~ {R_best.max():.1f}")
    print("-" * 50)
    print(f"Model Coefficients: W = A * EI^B * R^C + D")
    print(f"  A = {popt_best[0]:.10e}")
    print(f"  B = {popt_best[1]:.8f}")
    print(f"  C = {popt_best[2]:.8f}")
    print(f"  D = {popt_best[3]:.8f}")
    print(f"  E = {popt_best[4]:.8f}")
    print("-" * 50)
    print("Copy these coefficients to your design code.")
    print("-" * 50)

    # 4. グラフによる検証
    verifier = CriticalVerifier(EI_best, R_best, W_best, W_pred_best)
    verifier.plot_dashboard()

if __name__ == "__main__":
    analyze_pareto_model()