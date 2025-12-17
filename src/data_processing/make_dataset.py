import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import os
import sys
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.spar_calculator import SparCalculator
from src.core.brazier_calculator import BrazierCalculator

# ファイルパス設定
RAW_DATA_PATH = os.path.join("data", "interim", "raw_spar_dataset_brazier.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "pareto_spar_dataset_brazier.csv")

class DatasetGenerator:
    def __init__(self):
        self.spar_calc = SparCalculator()
        self.brazier_calc = BrazierCalculator(knockdown_factor=0.5)

    def calculate_full_spec(self, ply_counts, diameter):
        # 1. SparCalculator (厳密な重量・剛性)
        EI_kgf, Weight_kg_m, T_total = self.spar_calc.calculate_spec(ply_counts, diameter)
        EI_Nmm2 = EI_kgf * 9.80665 # 単位換算
        
        # 2. BrazierCalculator (積分法による座屈強度)
        Mcrit_Nmm = self.brazier_calc.calculate_max_moment_integral(
            diameter, 
            ply_counts, 
            self.spar_calc.ply_angles
        )
        return EI_Nmm2, Weight_kg_m, Mcrit_Nmm, T_total

def process_diameter_base_pair(D, base_ply, cap_options, generator):
    local_results = []
    
    fixed_glass_in = 1
    fixed_torque = 2
    fixed_glass_out = 1
    
    ply_counts = np.zeros(11, dtype=int)
    ply_counts[0] = fixed_glass_in
    ply_counts[1] = fixed_torque
    ply_counts[2] = base_ply
    ply_counts[10] = fixed_glass_out
    
    for cap_config in cap_options:
        for i, count in enumerate(cap_config):
            ply_counts[3+i] = count
            
        EI, W, Mcrit, T = generator.calculate_full_spec(ply_counts, D)
        
        local_results.append({
            "R": D,
            "EI": EI,
            "Mcrit": Mcrit,
            "Weight": W,
            "BasePly": base_ply,
            "Thickness": T
        })
        
    return local_results

def generate_and_filter_dataset():
    # ==========================================
    # 1. RAWデータの生成 or 読み込み
    # ==========================================
    if os.path.exists(RAW_DATA_PATH):
        print(f"Found existing raw data at: {RAW_DATA_PATH}")
        print("Skipping generation and loading raw data...")
        raw_df = pd.read_csv(RAW_DATA_PATH)
    else:
        print("Generating Dataset (Parallel Processing)...")
        generator = DatasetGenerator()
        
        diameters = np.arange(30.0, 131.0, 2.0)
        base_options = range(10)
        cap_options = list(itertools.product(range(3), repeat=7))
        
        print(f"Combinations: {len(diameters)} Dia x {len(base_options)} Base x {len(cap_options)} Caps")
        
        tasks = [(D, base_ply) for D in diameters for base_ply in base_options]
        
        results_nested = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_diameter_base_pair)(D, base, cap_options, generator) 
            for D, base in tasks
        )
        
        all_data = [item for sublist in results_nested for item in sublist]
        raw_df = pd.DataFrame(all_data)
        
        # RAWデータの保存
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        raw_df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Raw Data saved to: {RAW_DATA_PATH}")
        
    print(f"Total Data Points: {len(raw_df):,}")
    
    # ==========================================
    # 2. フィルタリング (ここだけやり直せる)
    # ==========================================
    print("Filtering Pareto Optimal Solutions...")
    
    # EI: Log空間でBinning
    raw_df['LogEI'] = np.log10(raw_df['EI'] + 1e-9)
    raw_df['EI_Bin'] = pd.cut(raw_df['LogEI'], bins=200)

    # Mcrit: Log空間でBinning (座屈強度も考慮)
    raw_df['LogMcrit'] = np.log10(raw_df['Mcrit'] + 1e-9)
    raw_df['Mcrit_Bin'] = pd.cut(raw_df['LogMcrit'], bins=100)
    
    # グルーピング: (直径, EIビン, Mcritビン) で最軽量を残す
    idx = raw_df.groupby(['R', 'EI_Bin', 'Mcrit_Bin'], observed=True)['Weight'].idxmin()
    pareto_df = raw_df.loc[idx.dropna()].copy()
    
    pareto_df = pareto_df.sort_values(['R', 'EI', 'Mcrit'])
    
    print(f"Compressed Dataset Size: {len(pareto_df):,} points")
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    pareto_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed Dataset saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    generate_and_filter_dataset()