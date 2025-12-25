import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
import os
import sys

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 物理モデルは一切変更せず、既存の計算機を使用
from src.core.spar_calculator import SparCalculator

# 保存パス
RAW_DATA_PATH = os.path.join("data", "interim", "raw_spar_dataset_si.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "pareto_spar_dataset_si.csv")

class DatasetGenerator:
    def __init__(self):
        self.spar_calc = SparCalculator()
        self.g = 9.80665

    def calculate_spec(self, ply_counts, diameter):
        # 1. 物理モデルによる厳密計算 (出力は kgf, kg/m, mm)
        EI_kgf, Weight_kg_m, T_total = self.spar_calc.calculate_spec(ply_counts, diameter)
        
        # 2. 単位変換 (kgf -> N)
        # ここでSI単位に統一することで、以降の全工程での混乱を防ぐ
        EI_Nmm2 = EI_kgf * self.g
        
        return EI_Nmm2, Weight_kg_m, T_total

def process_diameter_base_pair(D, base_ply, cap_options, generator):
    local_results = []
    
    # 積層構成ルール (変更なし)
    # [Glass, Torque, Base, Cap1...Cap7, Glass]
    ply_counts = np.zeros(11, dtype=int)
    ply_counts[0] = 1 # Glass In
    ply_counts[1] = 2 # Torque
    ply_counts[2] = base_ply # Base
    ply_counts[10] = 1 # Glass Out
    
    for cap_config in cap_options:
        # Cap層の設定
        for i, count in enumerate(cap_config):
            ply_counts[3+i] = count
            
        # 計算実行
        EI, W, T = generator.calculate_spec(ply_counts, D)
        
        local_results.append({
            "R": D,                # 直径 [mm]
            "EI": EI,              # 曲げ剛性 [N・mm^2] (SI単位)
            "Weight": W,           # 重量 [kg/m]
            "BasePly": base_ply,
            "Thickness": T
        })
        
    return local_results

def main():
    print("Generating SI-Unit Dataset...")
    generator = DatasetGenerator()
    
    # 探索範囲 (変更なし)
    diameters = np.arange(30.0, 131.0, 2.0)
    base_options = range(10)
    cap_options = list(itertools.product(range(3), repeat=7)) # 3^7 = 2187通り
    
    tasks = [(D, base_ply) for D in diameters for base_ply in base_options]
    
    print(f"Total Tasks: {len(tasks)} (Caps per task: {len(cap_options)})")

    # 並列処理
    results_nested = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_diameter_base_pair)(D, base, cap_options, generator) 
        for D, base in tasks
    )
    
    # フラット化
    all_data = [item for sublist in results_nested for item in sublist]
    raw_df = pd.DataFrame(all_data)
    
    # 生データ保存
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw data saved: {RAW_DATA_PATH}")

    # --- フィルタリング (パレート最適解) ---
    print("Filtering Pareto Optimal Solutions...")
    
    # 対数空間でのビニング処理
    raw_df['LogEI'] = np.log10(raw_df['EI'] + 1e-9)
    raw_df['EI_Bin'] = pd.cut(raw_df['LogEI'], bins=300)

    # (直径, 剛性ビン) の組み合わせで「最も軽い」個体のみ残す
    # これにより、同じ剛性なら軽い方が良いという物理的妥当性を担保
    idx = raw_df.groupby(['R', 'EI_Bin'], observed=True)['Weight'].idxmin()
    pareto_df = raw_df.loc[idx.dropna()].copy()
    
    pareto_df = pareto_df.sort_values(['R', 'EI'])
    pareto_df = pareto_df.drop(columns=['LogEI', 'EI_Bin']) # 不要列削除
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    pareto_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved: {PROCESSED_DATA_PATH}")
    print(f"Dataset Size: {len(pareto_df)}")

if __name__ == "__main__":
    main()