import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 物理モデル (SparCalculator) - SAME AS YOURS
# ==========================================
class SparCalculator:
    def __init__(self):
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

        # EI calculation
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

        # Weight calculation
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights_kg_m = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights_kg_m[i] = 0.001496 * rS[i]
            else:
                weights_kg_m[i] = 0.001559 * rS[i]
        
        return total_EI_Nmm2, np.sum(weights_kg_m)

# ==========================================
# 2. データ生成 & 解析
# ==========================================
def generate_large_dataset(n_samples=50000):
    calc = SparCalculator()
    data_EI, data_R, data_Weight = [], [], []
    
    print(f"Generating {n_samples} samples...")
    for _ in range(n_samples):
        R_dia = random.uniform(40.0, 120.0)
        current_plies = np.array([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        # Randomize middle layers slightly more realistically for wider range
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

# ==========================================
# 3. 批判的検証用クラス (NEW)
# ==========================================
class CriticalVerifier:
    def __init__(self, EI, R, W_actual, W_pred):
        self.EI = EI
        self.R = R
        self.W_actual = W_actual
        self.W_pred = W_pred
        
        # 相対誤差 (%)
        self.rel_error = (self.W_pred - self.W_actual) / self.W_actual * 100.0
        
    def plot_dashboard(self):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Critical Verification of Weight Estimation Model: $W = A \cdot EI^B \cdot R^C + D$", fontsize=16)

        # --- Plot 1: Actual vs Predicted (Accuracy) ---
        ax1 = fig.add_subplot(2, 3, 1)
        sc1 = ax1.scatter(self.W_actual, self.W_pred, c=np.abs(self.rel_error), cmap='coolwarm', s=5, alpha=0.6, vmin=0, vmax=5)
        ax1.plot([self.W_actual.min(), self.W_actual.max()], [self.W_actual.min(), self.W_actual.max()], 'k--', lw=1)
        ax1.set_xlabel('Actual Weight (kg/m)')
        ax1.set_ylabel('Predicted Weight (kg/m)')
        ax1.set_title('Accuracy Check (Color: Abs Error %)')
        plt.colorbar(sc1, ax=ax1, label='Abs Error %')
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Relative Error Histogram (Distribution) ---
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(self.rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Error Distribution\nMean: {np.mean(self.rel_error):.2f}%, Std: {np.std(self.rel_error):.2f}%')
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Residuals vs Diameter (Bias Check 1) ---
        ax3 = fig.add_subplot(2, 3, 4)
        sc3 = ax3.scatter(self.R, self.rel_error, c=self.EI, cmap='viridis', s=5, alpha=0.5)
        ax3.axhline(0, color='red', linestyle='--', lw=1)
        ax3.set_xlabel('Diameter R (mm)')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_title('Bias vs Diameter (Color: EI)')
        plt.colorbar(sc3, ax=ax3, label='EI (Nmm2)')
        ax3.grid(True, alpha=0.3)

        # --- Plot 4: Residuals vs EI (Bias Check 2) ---
        ax4 = fig.add_subplot(2, 3, 5)
        # Log scale for EI usually makes sense
        sc4 = ax4.scatter(self.EI, self.rel_error, c=self.R, cmap='plasma', s=5, alpha=0.5)
        ax4.axhline(0, color='red', linestyle='--', lw=1)
        ax4.set_xscale('log')
        ax4.set_xlabel('Stiffness EI (Nmm2) [Log]')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title('Bias vs Stiffness (Color: R)')
        plt.colorbar(sc4, ax=ax4, label='Diameter R')
        ax4.grid(True, alpha=0.3)

        # --- Plot 5: 3D Visualization of the Fit ---
        ax5 = fig.add_subplot(2, 3, (3, 6), projection='3d')
        # Subsample for 3D plot performance if data is huge
        idx_sub = np.random.choice(len(self.EI), min(2000, len(self.EI)), replace=False)
        
        p = ax5.scatter(np.log10(self.EI[idx_sub]), self.R[idx_sub], self.W_actual[idx_sub], 
                        c=self.rel_error[idx_sub], cmap='bwr', s=10, vmin=-5, vmax=5, label='Data')
        
        ax5.set_xlabel('Log10(EI)')
        ax5.set_ylabel('Diameter R')
        ax5.set_zlabel('Weight (kg/m)')
        ax5.set_title('3D Error Map (Red=Overpred, Blue=Underpred)')
        fig.colorbar(p, ax=ax5, label='Error %', shrink=0.6)

        plt.tight_layout()
        plt.show()

# ==========================================
# 4. Main Analysis
# ==========================================
def analyze_large_scale_model(top_percent=1.0): # Top 1% of lightest spars for given EI
    EI_raw, R_raw, W_raw = generate_large_dataset(n_samples=50000)
    
    # 1st Pass: Rough fit
    p0 = [1e-5, 0.8, -1.3, 0.0]
    try:
        popt_all, _ = curve_fit(model_func_2d, (EI_raw, R_raw), W_raw, p0=p0, maxfev=50000)
    except RuntimeError:
        print("Optimization failed on full dataset.")
        return

    # Filter: Select "Efficient" spars (negative residuals = lighter than predicted average)
    W_pred_all = model_func_2d((EI_raw, R_raw), *popt_all)
    residuals = W_raw - W_pred_all # Actual - Pred. Negative means Actual < Pred (Good)
    
    # We want the spars that are LIGHTER than the average trend
    cutoff_index = int(len(residuals) * (top_percent / 100.0))
    sorted_indices = np.argsort(residuals) # Smallest (most negative) first
    best_indices = sorted_indices[:cutoff_index]
    
    EI_best = EI_raw[best_indices]
    R_best = R_raw[best_indices]
    W_best = W_raw[best_indices]
    
    # 2nd Pass: Fit only on the "Best" designs
    popt_best, _ = curve_fit(model_func_2d, (EI_best, R_best), W_best, p0=popt_all, maxfev=50000)
    W_pred_best = model_func_2d((EI_best, R_best), *popt_best)

    print("-" * 40)
    print(f"Analysis on Top {top_percent}% Efficient Spars ({len(EI_best)} samples)")
    print(f"Optimized Parameters: W = A * EI^B * R^C + D")
    print(f"  A = {popt_best[0]:.6e}")
    print(f"  B = {popt_best[1]:.6f}")
    print(f"  C = {popt_best[2]:.6f}")
    print(f"  D = {popt_best[3]:.6f}")
    print("-" * 40)

    # --- CRITICAL VERIFICATION ---
    verifier = CriticalVerifier(EI_best, R_best, W_best, W_pred_best)
    verifier.plot_dashboard()

if __name__ == "__main__":
    analyze_large_scale_model(top_percent=1.0)