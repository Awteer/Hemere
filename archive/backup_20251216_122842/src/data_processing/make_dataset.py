import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# SparCalculatorはプロジェクト内のsrc.coreにあると仮定
from src.core.spar_calculator import SparCalculator

# ==========================================
# 1. 全探索データ生成 (Full Search)
# ==========================================
def generate_full_search_dataset():
    calc = SparCalculator()
    
    # 直径範囲: 30mm 〜 130mm (2mmステップで51点と仮定)
    diameters = np.arange(30.0, 131.0, 2.0)
    
    # 積層パターンの生成 (全探索)
    # Base層 (Index 2): 0 ~ 9枚
    base_options = range(10)
    # Cap層 (Index 3~9): 0 ~ 2枚 (3^7 = 2187通り)
    cap_options = list(itertools.product(range(3), repeat=7))
    
    total_combinations = len(diameters) * len(base_options) * len(cap_options)
    print(f"Generating Full Search Dataset...")
    print(f"Diameters: {len(diameters)} steps ({diameters.min()}-{diameters.max()}mm)")
    print(f"Combinations per diameter: {len(base_options) * len(cap_options):,} patterns")
    print(f"Total combinations to calculate: {total_combinations:,}")
    
    all_data = []

    for D in tqdm(diameters, desc="Processing Diameters"):
        
        ply_counts = np.zeros(11, dtype=int)
        ply_counts[0] = 1   # Glass
        ply_counts[10] = 1  # Glass
        ply_counts[1] = 2   # Torque (2枚固定)

        for base_ply in base_options:
            ply_counts[2] = base_ply
            
            for cap_config in cap_options:
                ply_counts[3:10] = cap_config
                
                # 計算
                EI, W, t_total = calc.calculate_spec(ply_counts, D)
                
                all_data.append({
                    "EI": EI,
                    "R": D,
                    "Weight": W,
                    "Thickness": t_total,
                    "PlyConfig": str(ply_counts.tolist())
                })
    
    df = pd.DataFrame(all_data)
    print(f"Total Generated Points: {len(df):,}")
    return df

# ==========================================
# 2. ビニング & フィルタリング (Linear / Equal Interval)
# ==========================================
def filter_best_dataset(df, bins=2000):
    """
    剛性(EI)を「対数」ではなく「等間隔(Linear)」で分割する。
    Rの多様性を維持するため Local Optimum (R x EI_Bin) を残す。
    """
    print(f"Filtering Local Pareto Frontier (Linear Binning, bins={bins})...")
    
    # 以前の対数変換 (np.log10) は廃止し、生のEIを使うことで高剛性域の解像度を確保
    
    # ビン分割 (pd.cut は線形分割を行う)
    df['EI_Bin'] = pd.cut(df['EI'], bins=bins)
    
    # 【Local Optimum 戦略】
    # 「ある剛性ビン(Bin) × ある直径(R)」の中で最も軽いものを残す
    idx = df.groupby(['R', 'EI_Bin'], observed=True)['Weight'].idxmin()
    
    best_df = df.loc[idx.dropna()].copy()
    
    # 学習用にLogEI列は作っておく（フィルタリングには使わない）
    best_df['LogEI'] = np.log10(best_df['EI'])
    
    # ソートして整える
    best_df = best_df.sort_values(['R', 'EI'])
    
    print(f"Compressed Dataset Size: {len(best_df):,}")
    return best_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. データ生成
    raw_df = generate_full_search_dataset()
    
    # 生データ側にも LogEI を計算しておく (描画用)
    raw_df['LogEI'] = np.log10(raw_df['EI'])
    
    # 2. フィルタリング (Linear Binning)
    # データ点の多様性と高解像度を両立するため、ビン数を2000に設定
    pareto_df = filter_best_dataset(raw_df, bins=2000)
    
    # 3. 保存
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "pareto_spar_dataset_eos_model.csv")
    pareto_df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path}")
    
    # 4. 可視化
    plt.figure(figsize=(12, 8))
    
    # 全データ（薄く背景に）
    sample_raw = raw_df.sample(min(20000, len(raw_df)))
    plt.scatter(sample_raw['LogEI'], sample_raw['Weight'], c='lightgray', s=1, alpha=0.3, label='Raw Search Space')
    
    # パレート解（メイン）
    sc = plt.scatter(
        pareto_df['LogEI'], 
        pareto_df['Weight'], 
        c=pareto_df['R'], 
        cmap='viridis', 
        s=15, 
        edgecolors='k', 
        linewidth=0.5,
        vmin=30, vmax=140,
        label='Pareto Frontier'
    )
    
    plt.colorbar(sc, label='Diameter R [mm]')
    plt.xlabel('Log10(EI) [kgf mm^2]')
    plt.ylabel('Weight [kg/m]')
    plt.title('Eos Model Training Data (Linear Binning / Local Pareto)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 画像保存
    img_path = os.path.join("results", "figures", "dataset_distribution.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    print(f"Plot saved to: {img_path}")
    
    # plt.show()