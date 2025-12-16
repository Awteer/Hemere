import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# データパス
CSV_PATH = os.path.join("data", "processed", "pareto_spar_dataset_eos_model.csv")

# =========================================================
# ユーザー様提供の分析関数 (そのまま採用)
# =========================================================
def analyze_error_distribution(df_input: pd.DataFrame):
    """
    EOS (Error of Specification) の分布を検証する関数
    剛性ターゲット(EI)と直径(Diameter)を軸に、剛性誤差と重量ペナルティをヒートマップで可視化する。
    """
    df = df_input.copy()

    # --- 1. 前処理: 剛性のBinning (Target_EIの作成) ---
    # LogスケールでBinを切ることで、広範囲の剛性を分析しやすくする
    n_bins = 50
    ei_min = df['EI'].min()
    ei_max = df['EI'].max()
    
    # log spaceでbinを作成
    bins = np.logspace(np.log10(ei_min), np.log10(ei_max), n_bins)
    
    # 各行がどのBinに属するか
    df['EI_Bin_Idx'] = np.digitize(df['EI'], bins)
    # Binの下限値をTarget_EIとする (最大インデックス超過回避)
    df['Target_EI'] = df['EI_Bin_Idx'].apply(lambda x: bins[min(x, len(bins)-1)])

    # --- 2. エラー指標の計算 ---

    # (A) 剛性乖離率 (Stiffness Overshoot Error)
    # 狙った剛性ターゲットに対して、実際どれだけ上振れしたか (離散積層によるペナルティ)
    df['Stiffness_Error_Pct'] = (df['EI'] - df['Target_EI']) / df['Target_EI'] * 100

    # (B) 重量ペナルティ (Weight Penalty vs Global Optimum)
    # その剛性Binの中で、「最も軽い個体」を基準とする
    
    # Binごとの最小重量を計算 (Global Optimum)
    global_min_weights = df.groupby('Target_EI')['Weight'].min().reset_index()
    global_min_weights.rename(columns={'Weight': 'Global_Min_Weight'}, inplace=True)
    
    df = pd.merge(df, global_min_weights, on='Target_EI', how='left')
    
    # 理想重量からの乖離率 (%)
    df['Weight_Penalty_Pct'] = (df['Weight'] - df['Global_Min_Weight']) / df['Global_Min_Weight'] * 100

    # --- 3. ピボットテーブルの作成 (Heatmap用) ---
    
    # Diameterカラム名を変更 (元のデータは'R')
    df.rename(columns={'R': 'Diameter'}, inplace=True)

    # 剛性乖離率 (平均)
    pivot_stiffness = df.pivot_table(
        index='Target_EI', 
        columns='Diameter', 
        values='Stiffness_Error_Pct', 
        aggfunc='mean'
    )
    
    # 重量ペナルティ (最小) - その条件における最良のペナルティを見る
    pivot_weight = df.pivot_table(
        index='Target_EI', 
        columns='Diameter', 
        values='Weight_Penalty_Pct', 
        aggfunc='min'
    )

    # --- 4. 可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ヒートマップ1: 剛性乖離 (Stiffness Error)
    sns.heatmap(pivot_stiffness.sort_index(ascending=False), ax=axes[0], cmap='viridis', 
                cbar_kws={'label': 'Stiffness Overshoot (%)'}, fmt=".1f")
    axes[0].set_title('Stiffness Error Distribution\n(High = Discrete Ply Limitations)')
    axes[0].set_ylabel(f'Target EI (Log10 Label: {np.log10(pivot_stiffness.index.min()):.1f} to {np.log10(pivot_stiffness.index.max()):.1f})')
    axes[0].set_xlabel('Mandrel Diameter (mm)')

    # ヒートマップ2: 重量ペナルティ (Weight Penalty)
    sns.heatmap(pivot_weight.sort_index(ascending=False), ax=axes[1], cmap='magma', 
                cbar_kws={'label': 'Weight Penalty vs Global Opt (%)'}, vmin=0, vmax=50, fmt=".1f") 
    axes[1].set_title('Weight Penalty Distribution\n(High = Bad Diameter Choice for this EI)')
    axes[1].set_ylabel('Target EI (Nmm^2)')
    axes[1].set_xlabel('Mandrel Diameter (mm)')

    plt.tight_layout()
    plt.show()

    # 画像保存
    save_path = os.path.join("results", "figures", "spar_design_space_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Analysis plot saved to: {save_path}")

    return df

# =========================================================
# メイン実行部
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}. Please run make_dataset.py first.")
    else:
        # フィルタリング済みの学習データセットを読み込む
        df_data = pd.read_csv(CSV_PATH)
        
        # 'Diameter'カラム名は'R'として保存されている前提
        required_cols = ['EI', 'Weight', 'R']
        if not all(col in df_data.columns for col in required_cols):
             print(f"Error: Dataset is missing required columns. Found: {df_data.columns.tolist()}")
        else:
            print(f"Analyzing {len(df_data):,} data points...")
            analyzed_df = analyze_error_distribution(df_data)

            print("\nAnalysis Complete. Check the generated heatmaps.")