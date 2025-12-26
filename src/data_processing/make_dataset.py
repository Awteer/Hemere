import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.spar_calculator import SparCalculator

# ==========================================
# 1. 全探索データ生成 (Full Search)
# ==========================================
def generate_full_search_dataset():
    calc = SparCalculator()
    
    # 直径範囲: 30mm 〜 130mm
    diameters = np.arange(30.0, 131.0, 2.0)
    
    # 積層パターンの生成 (一度だけ計算)
    # Base層 (Index 2): 0 ~ 9枚
    base_options = range(10)
    # Cap層 (Index 3~9): 0 ~ 2枚 (3^7 = 2187通り)
    cap_options = list(itertools.product(range(3), repeat=7))
    
    print(f"全積層構成を作成しています...")
    print(f"直径: {len(diameters)} ステップ:30-130mm)")
    print(f"直径毎に評価: {len(base_options) * len(cap_options):,}パターン")
    
    # 結果格納用リスト
    # メモリ節約のため、辞書リストではなくリストのリストを使う等の最適化も可能だが、
    # 今回は可読性重視で辞書リストを使用
    all_data = []

    # プログレスバーで進捗表示
    for D in tqdm(diameters, desc="Processing Diameters"):
        
        # 毎回インスタンス化せず、計算用の変数を使い回す
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
                
                # 結果追加 (軽量化のためPlyConfigは文字列化せず必要な時だけ復元する運用もアリだが、今回は含める)
                all_data.append({
                    "EI": EI,
                    "R": D,
                    "Weight": W,
                    "Thickness": t_total,
                    "PlyConfig": str(ply_counts.tolist())
                })
    
    df = pd.DataFrame(all_data)
    print(f"合計作成点数: {len(df):,}")
    return df

# ==========================================
# 2. ビニング & フィルタリング (Pareto Optimization)
# ==========================================
def filter_best_dataset(df, bins=300):
    """
    剛性(LogEI)で細かくビン分割し、
    【各直径(R) × 各剛性ビン】 の組み合わせにおける最軽量設計を残す。
    (Local Optimumを残す設定)
    """
    print("Filtering Local Pareto Frontier (Best Weight per R & Stiffness Bin)...")
    
    # EIの対数をとる
    df['LogEI'] = np.log10(df['EI'])
    
    # ビン分割
    df['EI_Bin'] = pd.cut(df['LogEI'], bins=bins)
    
    # 【修正箇所】
    # GroupByに 'R' を追加しました。
    # これで「ある剛性ビンの中で、直径R=30mmならこれがベスト」「R=31mmならこれがベスト」...
    # というデータが全て残ります。
    idx = df.groupby(['R', 'EI_Bin'], observed=True)['Weight'].idxmin()
    
    best_df = df.loc[idx.dropna()].copy()
    
    # 見やすいようにソート (直径順 -> 剛性順)
    best_df = best_df.sort_values(['R', 'EI'])
    
    print(f"Compressed Dataset Size: {len(best_df):,}")
    return best_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. データ生成
    raw_df = generate_full_search_dataset()
    
    # 2. フィルタリング
    # ビン数を増やすと、より滑らかな曲線としてデータが残る
    pareto_df = filter_best_dataset(raw_df, bins=500)
    
    # 3. 保存
    # 保存先ディレクトリ
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "pareto_spar_dataset_eos_model.csv")
    pareto_df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path}")
    
    # 4. 可視化 (スケール修正版)
    plt.figure(figsize=(12, 8))
    
    # 全データ（薄く背景に）- 重いので間引く
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
        vmin=30, vmax=140,  # ★カラースケールを固定
        label='Pareto Frontier'
    )
    
    plt.colorbar(sc, label='Diameter R [mm]')
    plt.xlabel('Log10(EI) [kgf mm^2]')
    plt.ylabel('Weight [kg/m]')
    plt.title('Eos Model Training Data (Global Pareto Frontier)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 画像保存
    img_path = os.path.join("results", "figures", "dataset_distribution.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    print(f"Plot saved to: {img_path}")
    
    plt.show() # 必要ならコメントアウト解除