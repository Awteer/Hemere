import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 

# ==========================================
# 1. 物理モデル (User Provided SparCalculator)
# ==========================================
class SparCalculator:
    def __init__(self):
        # Material Props in kgf/mm^2 
        # [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        # Indices: 0,      1,      2,      3,      4,      5
        self.mat_props_kgf = np.array([13000, 1900, 900, 22000, 1900, 800])
        
        # 層ごとの有効角度幅 (degree) と推測される
        # Idx 0,10: 24t (90deg)
        # Idx 1: 40t (+-45deg)
        # Idx 2: 40t (Φ90)
        # Idx 3~9: Caps (Φ50~Φ20)
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
        Ix = (outer**4 - inner**4) * (self.pp * np.pi /360 + np.sin(2*self.pp*np.pi/180)/4)/16
        
        E_vec_kgf = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                # 最内・最外: Mat index 2 (24t_90) -> E=900
                E_vec_kgf[i] = self.mat_props_kgf[2]
            elif idx == 2:
                # 2層目: Mat index 4 (40t_45) -> E=1900
                E_vec_kgf[i] = self.mat_props_kgf[4]
            else:
                # 3層目以降: Mat index 3 (40t_0) -> E=22000
                E_vec_kgf[i] = self.mat_props_kgf[3]
        
        EIx_kgf = Ix * E_vec_kgf
        total_EI_Nmm2 = np.sum(EIx_kgf) * self.g

        # Weight calculation (kg/m)
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        weights_kg_m = np.zeros(n_layers)
        
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights_kg_m[i] = 0.001496 * rS[i]
            else:
                weights_kg_m[i] = 0.001559 * rS[i]
        
        total_weight_kg_m = np.sum(weights_kg_m)
        
        # D/tチェック用に総厚みも計算して返す
        total_thickness_mm = np.sum(thickness)

        return total_EI_Nmm2, total_weight_kg_m, total_thickness_mm

# ==========================================
# 2. データ生成ロジック (改良版: Max 2ply制約対応)
# ==========================================
def generate_optimized_dataset():
    calc = SparCalculator()
    data_list = []
    
    # パラメータ設定
    diameters = np.arange(30.0, 141.0, 1.0) 
    
    # ここを修正: ただ回すだけでなく「戦略」を分ける
    print(f"Generating comprehensive dataset...")

    for D in tqdm(diameters):
        # -------------------------------------------------
        # 戦略A: ミニマムゲージ（極薄）作成 ★ここが最重要
        # 大口径・低剛性 (Left-Top領域) を埋めるためのデータ
        # -------------------------------------------------
        for _ in range(500): # 各直径で50個くらい作れば十分
            ply_counts = np.zeros(11)
            # 最内・最外・トルク層のみ（これが製造可能な最小構成と仮定）
            ply_counts[0] = 1   # Glass
            ply_counts[10] = 1  # Glass
            ply_counts[1] = 2   # Torque
            
            # UD材（Index 2~9）を「ほぼゼロ」にする
            # 0〜1プライを稀に追加する程度
            ply_counts[2] = np.random.randint(0, 2) # Base 0度
            
            # キャップ層はほぼ積まない
            caps = np.zeros(7)
            # たまに薄く1枚だけ積む（ノイズを与える）
            if np.random.rand() > 0.5:
                idx = np.random.randint(0, 7)
                caps[idx] = 1
            
            ply_counts[3:10] = caps
            
            # 計算・保存
            EI, W, t_total = calc.calculate_spec(ply_counts, D)
            if (D / t_total) <= 250.0: # 薄肉は大径だと座屈しやすいので、D/t制約は緩めるか外してAIに学習させる
                 data_list.append({"EI": EI, "R": D, "Weight": W, "Thickness": t_total})

        # -------------------------------------------------
        # 戦略B: ランダム生成（従来どおり）
        # バナナ型のメインストリームを作る
        # -------------------------------------------------
        for _ in range(1000): # 回数は適宜調整
            ply_counts = np.zeros(11)
            ply_counts[0] = 1
            ply_counts[10] = 1
            ply_counts[1] = 2
            ply_counts[2] = np.random.randint(0, 5) # Base
            
            # ランダム
            caps = np.random.randint(0, 3, size=7)
            ply_counts[3:10] = caps
            
            EI, W, t_total = calc.calculate_spec(ply_counts, D)
            if (D / t_total) <= 150.0: # 通常の制約
                 data_list.append({"EI": EI, "R": D, "Weight": W, "Thickness": t_total})
                 
        # -------------------------------------------------
        # 戦略C: マックスゲージ（厚肉）作成
        # 小径・高剛性 (Bottom-Right領域) を埋める
        # -------------------------------------------------
        for _ in range(50):
            ply_counts = np.zeros(11)
            ply_counts[0] = 1
            ply_counts[10] = 1
            ply_counts[1] = 2
            ply_counts[2] = 4 # Max
            ply_counts[3:10] = 2 # Max Cap
            
            EI, W, t_total = calc.calculate_spec(ply_counts, D)
            data_list.append({"EI": EI, "R": D, "Weight": W, "Thickness": t_total})

    df = pd.DataFrame(data_list)
    print(f"Raw dataset size: {len(df)}")
    
    # 重複削除などはしてもいいが、AI学習用なら生データのままでもOK
    return df

# ==========================================
# 3. パレートフィルタリング
# ==========================================
def filter_pareto_frontier(df):
    print("Filtering Pareto frontier (Strict Dominated Sort)...")
    pareto_data = []

    # 直径ごとにグループ化
    for r_val, group in tqdm(df.groupby('R')):
        # 剛性 (EI) が高い順にソート
        sorted_group = group.sort_values('EI', ascending=False)
        
        current_min_weight = float('inf')
        
        # 上から見ていき、重量記録を更新できた設計のみ採用
        for _, row in sorted_group.iterrows():
            if row['Weight'] < current_min_weight:
                pareto_data.append(row)
                current_min_weight = row['Weight']

    best_df = pd.DataFrame(pareto_data)
    print(f"Pareto optimized dataset size: {len(best_df)}")
    return best_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. データ生成
    raw_df = generate_optimized_dataset()
    
    # 2. パレート解抽出
    pareto_df = filter_pareto_frontier(raw_df)
    
    # 3. 保存
    csv_name = "pareto_spar_dataset_user_model.csv"
    pareto_df.to_csv(csv_name, index=False)
    #raw_df.to_csv(csv_name, index=False)
    print(f"Dataset saved to {csv_name}")
    
    # 4. 分布確認 (可視化)
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(pareto_df['EI']), pareto_df['Weight'], 
                c=pareto_df['R'], cmap='viridis', s=3, alpha=0.8)

    plt.colorbar(label='Diameter R (mm)')
    plt.xlabel('Log10(EI) [Nmm^2]')
    plt.ylabel('Weight [kg/m]')
    plt.title('Pareto Frontier using User SparCalculator')
    plt.grid(True, alpha=0.3)
    plt.show()