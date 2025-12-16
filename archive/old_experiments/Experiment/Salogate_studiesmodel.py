import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from tqdm import tqdm # 進捗バー表示用 (pip install tqdm)

# ==========================================
# 1. マテリアル定義 (拡張性を高める)
# ==========================================
@dataclass
class Material:
    name: str
    E1: float  # 繊維方向ヤング率 [kgf/mm2]
    E2: float  # 直交方向ヤング率 [kgf/mm2]
    rho: float # 密度 [g/cm3] -> 計算内で単位合わせる
    t: float   # 1層厚み [mm]

# 代表的な材料 (HPAでよく使われる物性値に調整してください)
# ここでは例としてTORAY系やGlassを想定
MAT_DB = {
    "HrC": Material("HighModulus_CF", 40000, 1000, 1.60, 0.100), # M40J級
    "StC": Material("Standard_CF",    24000, 1500, 1.55, 0.120), # T700級
    "GF":  Material("GlassFiber",      4000, 4000, 2.00, 0.050)  # 極薄ガラスクロス
}

# ==========================================
# 2. 高機能スパ計算クラス
# ==========================================
class AdvancedSparCalculator:
    def __init__(self):
        self.g = 9.80665

    def calc_stiffness_and_weight(self, diameter, layer_stack):
        """
        layer_stack: List of tuples [(Material_obj, angle_deg), ...]
        積層順序: リストの0番目が「最内層」、最後が「最外層」とする
        """
        # 半径計算の準備
        current_r = diameter / 2.0
        
        total_EI = 0.0
        total_area_density = 0.0 # g/mm length -> kg/m
        
        # 積層理論に基づく計算 (薄肉円筒近似ではなく、層ごとの積分を行う)
        for mat, angle in layer_stack:
            r_inner = current_r
            r_outer = current_r + mat.t
            
            # 断面二次モーメント Ix の寄与 (薄肉円環近似)
            # Ix = pi * r^3 * t
            r_mid = (r_inner + r_outer) / 2.0
            I_layer = np.pi * (r_mid**3) * mat.t
            
            # 複合則 (配向角考慮)
            # 簡易式: Ex = E1*cos^4(t) + E2*sin^4(t) ... 厳密にはQマトリクスだが、
            # HPAレベルの概算なら cos^2 or cos^4 則で近似運用することも多い。
            # ここでは一般的な設計シートに合わせて「cos^4近似」または「異方性考慮」
            # ※前回のコードに合わせて簡易的に実装します
            
            rad = np.deg2rad(angle)
            # 簡易的なExの算出 (Classical Laminate Theory簡易版)
            # 0度方向への寄与率
            Ex = mat.E1 * (np.cos(rad)**4) + mat.E2 * (np.sin(rad)**4) 
            # ※本来はG12なども絡みますが、剛性寄与の9割は0度層なので一旦これで走らせます
            
            EI_layer = Ex * I_layer
            total_EI += EI_layer
            
            # 重量計算 (密度 * 体積)
            # 断面積 A = 2 * pi * r_mid * t
            area = 2 * np.pi * r_mid * mat.t
            # weight per length (g/mm = kg/m)
            w_per_len = (area * mat.rho / 1000.0) # g/mm -> kg/m 換算
            # mat.rho is g/cm3 = mg/mm3. 
            # area(mm2) * rho(mg/mm3) = mg/mm = g/m
            # kg/m = g/m / 1000
            
            # 修正: rho(g/cm3) = rho * 1e-3 (g/mm3)
            # w (g/mm) = area(mm2) * rho(g/cm3)*1e-3
            # w (kg/m) = w(g/mm) * 1 = area * rho * 1e-3
            
            w_unit = area * (mat.rho * 1e-3)
            total_area_density += w_unit
            
            current_r = r_outer

        # kgf*mm2 -> N*mm2
        return total_EI * self.g, total_area_density

# ==========================================
# 3. データセット生成ロジック (組み合わせ爆発を制御)
# ==========================================
def generate_comprehensive_dataset():
    calc = AdvancedSparCalculator()
    data_list = []

    # --- パラメータ設定 ---
    diameters = np.arange(40.0, 140.0, 2.0) # 40mm ~ 140mm (2mm刻み)
    
    # 積層パターンの生成
    # ルール: 
    # 1. 最内層は必ずGFかクロス (離型性/座屈) -> 今回は計算簡略化のため省略可だが一応入れる
    # 2. 0度は主強度部材。±45度はねじり/座屈対策。
    # 3. 積層構成の「テンプレート」を用意して、枚数を振る
    
    # テンプレート: (材料キー, 角度)
    # 高弾性0度をメインに、45度を補強に入れるパターン
    
    print("Generating comprehensive dataset...")
    
    # 積層枚数の組み合わせ (Iterator)
    # 0度層(Main): 2枚~10枚 (偶数推奨だが奇数も許容)
    # 45度層(Torque): 1枚~4枚
    # 補強0度(Stiff): 0枚~4枚 (部分積層を全周換算したイメージ)
    
    ranges = [
        range(2, 16), # Main 0deg layers
        range(1, 6),  # 45deg layers
        range(0, 5)   # Extra 0deg layers
    ]
    
    # 進捗表示用
    total_iters = len(diameters) * len(ranges[0]) * len(ranges[1]) * len(ranges[2])
    pbar = tqdm(total=total_iters)

    for D in diameters:
        for n_main, n_45, n_sub in itertools.product(*ranges):
            
            # 積層リストの構築
            stack = []
            
            # 1. 内層ガラス (オプション)
            stack.append((MAT_DB["GF"], 90)) 
            
            # 2. 45度層 (半分を+45, 半分を-45とするが剛性計算上は同じ)
            for _ in range(n_45):
                stack.append((MAT_DB["StC"], 45))
            
            # 3. メイン0度層 (高弾性)
            for _ in range(n_main):
                stack.append((MAT_DB["HrC"], 0))
                
            # 4. サブ0度層 (標準弾性 or 部分積層相当)
            for _ in range(n_sub):
                stack.append((MAT_DB["StC"], 0))
            
            # 5. 外層ガラス (保護)
            stack.append((MAT_DB["GF"], 90))
            
            # 計算
            EI, W = calc.calc_stiffness_and_weight(D, stack)
            
            # データを登録 (特徴量として Ply数も含めておく)
            # EI, R, W 以外にも、後で解析できるように内訳を残す
            data_list.append({
                "EI": EI,
                "LogEI": np.log10(EI),
                "R": D,
                "Weight": W,
                "n_0": n_main + n_sub, # トータル0度数
                "n_45": n_45
            })
            pbar.update(1)
            
    pbar.close()
    
    df = pd.DataFrame(data_list)
    print(f"Raw dataset size: {len(df)}")
    return df

# ==========================================
# 4. パレートフィルタリング (最強のデータのみ残す)
# ==========================================
def filter_pareto_frontier(df):
    print("Filtering Pareto frontier...")
    # 方法: LogEI と R でグリッドを切り、各グリッドで最小重量を抽出
    
    # ビン分割
    df['EI_bin'] = pd.cut(df['LogEI'], bins=200)
    df['R_bin']  = pd.cut(df['R'], bins=50)
    
    # グループ化して最小重量のインデックスを取得
    # idxmin() は最小値を持つ行のIndexを返す
    idx = df.groupby(['EI_bin', 'R_bin'], observed=True)['Weight'].idxmin()
    
    # NaNを除去して抽出
    best_df = df.loc[idx.dropna()].reset_index(drop=True)
    
    print(f"Pareto optimized dataset size: {len(best_df)}")
    return best_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. 生成
    raw_df = generate_comprehensive_dataset()
    
    # 2. フィルタリング (重い設計は捨てる)
    pareto_df = filter_pareto_frontier(raw_df)
    
    # 3. 保存
    csv_name = "pareto_spar_dataset.csv"
    pareto_df.to_csv(csv_name, index=False)
    print(f"Dataset saved to {csv_name}")
    
    # 4. 可視化 (確認用)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_df['LogEI'], pareto_df['Weight'], c=pareto_df['R'], cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label='Diameter (mm)')
    plt.xlabel('Log10(EI)')
    plt.ylabel('Weight (kg/m)')
    plt.title('Generated Pareto Frontier for AI Training')
    plt.grid(True, alpha=0.3)
    plt.show()