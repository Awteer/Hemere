import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

# ==========================================
# 1. 物理モデル (SparCalculator) - FIXED
# ==========================================
class SparCalculator:
    def __init__(self):
        # Material Props in kgf/mm^2 (Original MATLAB values)
        # [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        self.mat_props_kgf = np.array([13000, 1900, 900, 22000, 1900, 800])
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])
        self.g = 9.80665

    def calculate_spec(self, ply_counts, R_diameter):
        n_layers = 11
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

        # EI calculation (kgf*mm^2 -> N*mm^2)
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
        total_EI_kgf = np.sum(EIx_kgf)
        total_EI_Nmm2 = total_EI_kgf * self.g # Convert to Newton

        # Weight calculation (FIXED)
        # Density factor from MATLAB: 0.001559 (g/mm^3 ?) 
        # Actually MATLAB logic: weight(g/m) = 1.559 * Area(mm^2)
        # So weight(kg/m) = 0.001559 * Area(mm^2)
        
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights_kg_m = np.zeros(n_layers)
        
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                # 0.001496 * Area gives kg/m
                weights_kg_m[i] = 0.001496 * rS[i]
            else:
                weights_kg_m[i] = 0.001559 * rS[i]
        
        total_weight_kg_m = np.sum(weights_kg_m)
        
        return total_EI_Nmm2, total_weight_kg_m

# ==========================================
# 2. データ生成 & 解析 (変更なし)
# ==========================================
def generate_large_dataset(n_samples=50000):
    calc = SparCalculator()
    data_EI = []
    data_R = []
    data_Weight = []
    
    print(f"Generating {n_samples} samples (FIXED Units)...")
    
    for _ in range(n_samples):
        R_dia = random.uniform(40.0, 120.0)
        current_plies = np.array([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        for k in range(2, 8):
            current_plies[k] = random.randint(0, 25)
        
        EI, W = calc.calculate_spec(current_plies, R_dia)
        data_EI.append(EI)
        data_R.append(R_dia)
        data_Weight.append(W)

    return np.array(data_EI), np.array(data_R), np.array(data_Weight)

def model_func_2d(X, a, b, c, d):
    ei, r = X
    return a * (ei ** b) * (r ** c) + d

def analyze_large_scale_model(top_percent=10.0):
    EI_raw, R_raw, W_raw = generate_large_dataset(n_samples=50000)
    
    p0 = [1e-5, 0.8, -1.3, 0.0]
    popt_all, _ = curve_fit(model_func_2d, (EI_raw, R_raw), W_raw, p0=p0, maxfev=50000)

    W_pred_all = model_func_2d((EI_raw, R_raw), *popt_all)
    residuals = W_raw - W_pred_all
    cutoff_index = int(len(residuals) * (top_percent / 100.0))
    sorted_indices = np.argsort(residuals)
    best_indices = sorted_indices[:cutoff_index]
    
    EI_best = EI_raw[best_indices]
    R_best = R_raw[best_indices]
    W_best = W_raw[best_indices]
    
    popt_best, _ = curve_fit(model_func_2d, (EI_best, R_best), W_best, p0=popt_all, maxfev=50000)
    
    print("-" * 40)
    print(f"Dataset Range (Check):")
    print(f"  EI (N*mm^2): {EI_best.min():.2e} ~ {EI_best.max():.2e}")
    print(f"  Diameter R : {R_best.min():.1f} ~ {R_best.max():.1f}")
    print(f"  Weight (kg/m): {W_best.min():.4f} ~ {W_best.max():.4f}")
    print("-" * 40)
    print(f"Optimized Parameters:")
    print(f"  A = {popt_best[0]:.6e}")
    print(f"  B = {popt_best[1]:.6f}")
    print(f"  C = {popt_best[2]:.6f}")
    print(f"  D = {popt_best[3]:.6f}")

    # Plot (略)
    plt.scatter(W_best, model_func_2d((EI_best, R_best), *popt_best), alpha=0.1, s=1)
    plt.xlabel('Actual Weight'); plt.ylabel('Predicted'); plt.show()

if __name__ == "__main__":
    analyze_large_scale_model(top_percent=1.0) # Top 1%で厳選