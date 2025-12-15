import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # 進捗バー表示用 (pip install tqdm)

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
        # Idx 0,10: Glass? (90)
        # Idx 1: Torque? (90)
        # Idx 2: Base 0? (90)
        # Idx 3~9: Caps (50~20)
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
        # Ixの計算式において、(1 - cos) の項が「部分積層」の効果を表している
        Ix = (np.pi / 64) * (outer**4 - inner**4) * (1 - np.cos(np.deg2rad(self.pp)))
        
        E_vec_kgf = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                # 最内・最外: Mat index 2 (24t_90 or Glass) -> E=900
                E_vec_kgf[i] = self.mat_props_kgf[2]
            elif idx == 2:
                # 2層目: Mat index 4 (40t_45 or 24t_45?) -> E=1900 (Torque layer)
                E_vec_kgf[i] = self.mat_props_kgf[4]
            else:
                # 3層目以降: Mat index 3 (40t_0) -> E=22000 (Main bending layer)
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
    samples_per_diameter = 50000 
    
    print(f"Generating dataset using user's SparCalculator (Max 2ply constraint)...")

    for D in tqdm(diameters):
        for _ in range(samples_per_diameter):
            # 積層構成 (ply_counts) の生成 [11層]
            ply_counts = np.zeros(11)
            
            # 1. 最内・最外 (Glass) -> 1枚固定
            ply_counts[0] = 1
            ply_counts[10] = 1
            
            # 2. トルク層 (Idx 1) -> 1〜3枚
            ply_counts[1] = np.random.randint(1, 4)
            
            # 3. ベース0度 (Idx 2) -> 0〜4枚
            ply_counts[2] = np.random.randint(0, 5)
            
            # 4. 集中積層キャップ (Idx 3~9: width 50~20) 
            # 【変更点】 各層 Max 2ply の制約下で生成
            
            # 戦略をランダムに切り替えることで、多様かつ「上手な」データを網羅する
            strategy = np.random.rand()
            
            if strategy < 0.7:
                # --- 戦略A: 完全ランダム探索 (70%) ---
                # 各層に 0, 1, 2枚 をランダムに割り当てる
                # 例: [2, 0, 1, 2, 0, 1, 0] ... スキップする構成も探索できる
                caps = np.random.randint(0, 3, size=7)
                
            else:
                # --- 戦略B: テーパー積層 (30%) ---
                # 「合計でN枚積みたい」とした時、幅の広い方(Index 3)から優先的に埋める
                # 人間がやる「Φ50, Φ50, Φ45, Φ45...」のような綺麗な段落ち設計を模倣
                
                # 目標合計プライ数 (0 ~ 14枚)
                target_total_plies = np.random.randint(0, 15)
                caps = np.zeros(7)
                
                # 幅広(Idx 3)から順に 2枚ずつ埋めていく
                remaining = target_total_plies
                for i in range(7):
                    if remaining >= 2:
                        caps[i] = 2
                        remaining -= 2
                    elif remaining == 1:
                        caps[i] = 1
                        remaining -= 1
                    else:
                        caps[i] = 0
            
            # 生成したcapsを適用
            ply_counts[3:10] = caps
            
            # --- 計算 ---
            EI, W, t_total = calc.calculate_spec(ply_counts, D)
            
            # --- 制約条件チェック ---
            if (D / t_total) <= 100.0:
                data_list.append({
                    "EI": EI,
                    "R": D,
                    "Weight": W,
                    "Thickness": t_total,
                    "PlyConfig": str(ply_counts.astype(int).tolist())
                })

    df = pd.DataFrame(data_list)
    print(f"Raw valid designs: {len(df)}")
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