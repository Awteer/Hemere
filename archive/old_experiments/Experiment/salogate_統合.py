import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import joblib
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D

# AI / Scikit-Learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 進捗バー (インストールされていれば使用)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# ==========================================
# 1. マテリアル & 物理モデル
# ==========================================
@dataclass
class Material:
    name: str
    E1: float  # 繊維方向ヤング率 [kgf/mm2]
    E2: float  # 直交方向ヤング率 [kgf/mm2]
    rho: float # 密度 [g/cm3]
    t: float   # 1層厚み [mm]

# 材料データベース
MAT_DB = {
    "HrC": Material("HighModulus_CF", 40000, 1000, 1.60, 0.100), # M40J級
    "StC": Material("Standard_CF",    24000, 1500, 1.55, 0.120), # T700級
    "GF":  Material("GlassFiber",      4000, 4000, 2.00, 0.050)  # 極薄ガラス
}

class AdvancedSparCalculator:
    def __init__(self):
        self.g = 9.80665

    def calc_spec(self, diameter, layer_stack):
        """
        Return: (EI [Nmm2], Weight [kg/m], TotalThickness [mm])
        """
        current_r = diameter / 2.0
        total_EI = 0.0
        total_area_density = 0.0
        total_thickness = 0.0 # 厚み計算用
        
        for mat, angle in layer_stack:
            # 厚み加算
            total_thickness += mat.t

            r_inner = current_r
            r_outer = current_r + mat.t
            r_mid = (r_inner + r_outer) / 2.0
            
            # 断面二次モーメント Ix
            I_layer = np.pi * (r_mid**3) * mat.t
            
            # 簡易複合則 (cos^4則)
            rad = np.deg2rad(angle)
            Ex = mat.E1 * (np.cos(rad)**4) + mat.E2 * (np.sin(rad)**4)
            
            EI_layer = Ex * I_layer
            total_EI += EI_layer
            
            # 重量計算 (kg/m)
            area = 2 * np.pi * r_mid * mat.t
            w_unit = area * (mat.rho * 1e-3)
            total_area_density += w_unit
            
            current_r = r_outer

        return total_EI * self.g, total_area_density, total_thickness

# ==========================================
# 2. データ生成 & フィルタリング (D/t制約付き)
# ==========================================
def generate_and_filter_data():
    calc = AdvancedSparCalculator()
    data_list = []

    # --- パラメータ探索範囲 ---
    # 直径: 30mm ~ 140mm (範囲を少し広げて探索)
    diameters = np.arange(30.0, 141.0, 1.0) 
    
    # 積層構成の探索
    # D/t制約が入るので、薄い積層は自動的に消える。
    # そのため、探索範囲は広めにとっておくのが安全。
    ranges = [
        range(2, 16), # Main (0deg): 2~15枚
        range(1, 6),  # Torque(45deg): 1~5枚
        range(0, 5)   # Sub (0deg): 0~4枚
    ]
    
    print("Generating dataset with D/t <= 100 constraint...")
    
    combinations = list(itertools.product(diameters, *ranges))
    
    # 除外された数カウント用
    rejected_count = 0

    for D, n_main, n_45, n_sub in tqdm(combinations):
        stack = []
        stack.append((MAT_DB["GF"], 90)) 
        for _ in range(n_45): stack.append((MAT_DB["StC"], 45))
        for _ in range(n_main): stack.append((MAT_DB["HrC"], 0))
        for _ in range(n_sub): stack.append((MAT_DB["StC"], 0))
        stack.append((MAT_DB["GF"], 90))
        
        # スペック計算
        EI, W, t_total = calc.calc_spec(D, stack)
        
        # ==========================================
        # 【重要】 D/t <= 100 チェック
        # ==========================================
        if (D / t_total) > 100.0:
            # 座屈リスク大のため採用しない（リストに入れない）
            rejected_count += 1
            continue

        data_list.append({
            "EI": EI,
            "R": D,
            "Weight": W,
            "Thickness": t_total,
            "D_t_Ratio": D / t_total
        })

    df = pd.DataFrame(data_list)
    print(f"Raw dataset size: {len(df)}")
    print(f"Rejected designs (Buckling Risk D/t > 100): {rejected_count}")

    # ==========================================
    # パレート解抽出 (Dominated Sort)
    # ==========================================
    print("Filtering Pareto frontier (Strict Dominated Sort)...")
    pareto_data = []

    for r_val, group in df.groupby('R'):
        # 剛性が高い順にソート
        sorted_group = group.sort_values('EI', ascending=False)
        current_min_weight = float('inf')
        
        for _, row in sorted_group.iterrows():
            # より軽ければ採用
            if row['Weight'] < current_min_weight:
                pareto_data.append(row)
                current_min_weight = row['Weight']

    best_df = pd.DataFrame(pareto_data)
    print(f"Pareto optimized dataset size: {len(best_df)}")
    
    return best_df['EI'].values, best_df['R'].values, best_df['Weight'].values

# ==========================================
# 3. 検証用クラス
# ==========================================
class CriticalVerifier:
    def __init__(self, EI, R, W_actual, W_pred):
        self.EI = EI
        self.R = R
        self.W_actual = W_actual
        self.W_pred = W_pred
        self.rel_error = (self.W_pred - self.W_actual) / (self.W_actual + 1e-9) * 100.0
        
    def plot_dashboard(self):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Verification: Random Forest Model (Constraint: D/t <= 100)", fontsize=16)

        # 1. Accuracy
        ax1 = fig.add_subplot(2, 3, 1)
        sc1 = ax1.scatter(self.W_actual, self.W_pred, c=np.abs(self.rel_error), cmap='coolwarm', s=10, alpha=0.8, vmin=0, vmax=2)
        ax1.plot([self.W_actual.min(), self.W_actual.max()], [self.W_actual.min(), self.W_actual.max()], 'k--', lw=1)
        ax1.set_xlabel('Actual Weight (kg/m)')
        ax1.set_ylabel('Predicted Weight (kg/m)')
        ax1.set_title('Accuracy Check')
        plt.colorbar(sc1, ax=ax1, label='Abs Error %')
        ax1.grid(True, alpha=0.3)

        # 2. Error Distribution
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(self.rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title(f'Error Distribution\nMean: {np.mean(self.rel_error):.3f}%, Std: {np.std(self.rel_error):.3f}%')
        ax2.grid(True, alpha=0.3)

        # 3. 3D Surface
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        idx_sub = np.random.choice(len(self.EI), min(2000, len(self.EI)), replace=False)
        p = ax3.scatter(np.log10(self.EI[idx_sub]), self.R[idx_sub], self.W_actual[idx_sub], 
                        c=self.rel_error[idx_sub], cmap='bwr', s=5, vmin=-1, vmax=1)
        ax3.set_xlabel('Log10(EI)')
        ax3.set_ylabel('Diameter R')
        ax3.set_zlabel('Weight')
        ax3.set_title('3D Pareto Surface')
        fig.colorbar(p, ax=ax3, label='Error %', shrink=0.6)

        # 4. Bias vs Diameter
        ax4 = fig.add_subplot(2, 3, 4)
        sc4 = ax4.scatter(self.R, self.rel_error, c=np.log10(self.EI), cmap='viridis', s=10, alpha=0.6)
        ax4.axhline(0, color='red', linestyle='--', lw=1)
        ax4.set_xlabel('Diameter R (mm)')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title('Bias vs Diameter')
        plt.colorbar(sc4, ax=ax4, label='Log10(EI)')
        ax4.grid(True, alpha=0.3)

        # 5. Bias vs Stiffness
        ax5 = fig.add_subplot(2, 3, 5)
        sc5 = ax5.scatter(self.EI, self.rel_error, c=self.R, cmap='plasma', s=10, alpha=0.6)
        ax5.axhline(0, color='red', linestyle='--', lw=1)
        ax5.set_xscale('log')
        ax5.set_xlabel('Stiffness EI (Nmm2) [Log]')
        ax5.set_title('Bias vs Stiffness')
        plt.colorbar(sc5, ax=ax5, label='R (mm)')
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ==========================================
# 4. メイン実行処理
# ==========================================
def run_ai_simulation_rf():
    # 1. データ生成 (D/t <= 100)
    EI_best, R_best, W_best = generate_and_filter_data()

    # --- 特徴量エンジニアリング ---
    log_EI = np.log10(EI_best)
    X = np.column_stack((log_EI, R_best))
    y = W_best

    # --- Random Forest モデル ---
    model_pipeline = Pipeline([
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("-" * 50)
    print("Training Random Forest Surrogate Model...")
    
    # 学習実行
    model_pipeline.fit(X, y)
    
    # 予測実行 (Validation)
    W_pred = model_pipeline.predict(X)
    
    # スコア計算
    score = r2_score(y, W_pred)
    mae = np.mean(np.abs(W_pred - y))
    
    print("-" * 50)
    print(f"Model Training Completed.")
    print(f"R^2 Score : {score:.6f}")
    print(f"MAE       : {mae:.5f} kg/m")
    
    # --- モデル保存 ---
    model_filename = 'spar_rf_model_safe.pkl'
    joblib.dump(model_pipeline, model_filename)
    
    print(f"Model saved to: {model_filename}")
    print("-" * 50)

    # 検証プロット
    verifier = CriticalVerifier(EI_best, R_best, W_best, W_pred)
    verifier.plot_dashboard()

if __name__ == "__main__":
    run_ai_simulation_rf()