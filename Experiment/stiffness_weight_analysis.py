import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools

# ==========================================
# 1. MATLABロジックの移植 (Core Physics)
# ==========================================

class SparCalculator:
    def __init__(self):
        # 定数定義 (MATLABコードより)
        # Material: [24t_0, 24t_45, 24t_90, 40t_0, 40t_45, 40t_90]
        # EI.mのMaterial配列: [13000 1900 900 22000 1900 800]
        # MATLAB index mapping: 
        # i=1,11 -> Material[2] (900) -> 24t_90 ?? (コードのロジック通りに実装)
        # i=2    -> Material[4] (1900) -> 24t_45 ??
        # else   -> Material[3] (22000) -> 40t_0 ??
        
        # ※注: MATLABコードのインデックスと値を忠実に再現します
        self.mat_props = np.array([13000, 1900, 900, 22000, 1900, 800])
        
        # 各層の配向角 (Enumerate_EI_Weight.m より)
        self.pp = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])

    def calculate_spec(self, ply_counts, R):
        """
        ply_counts: 各層のプライ数 (サイズ11の配列)
        R: マンドレル半径 (mm)
        Returns: (EI, weight_per_m)
        """
        # 層数
        n_layers = 11
        
        # 厚み計算
        thickness = ply_counts * 0.111
        thickness[0] = 0.125  # 1層目
        thickness[10] = 0.125 # 11層目
        
        # 内径・外径計算 (mm)
        inner = np.zeros(n_layers)
        outer = np.zeros(n_layers)
        
        current_inner = R
        inner[0] = current_inner
        
        # 1層目の外径
        outer[0] = inner[0] + 2 * thickness[0]
        
        for i in range(1, n_layers):
            inner[i] = inner[i-1] + 2 * thickness[i-1]
            # Outerのロジック (MATLAB: Outer(1)=Inner(2) ...あれ？)
            # MATLABコード: Outer(i) = Inner(i) + 2*thickness(i) が基本
            outer[i] = inner[i] + 2 * thickness[i]

        # --- EI計算 (EI.m) ---
        # 断面二次モーメント Ix
        # MATLAB: pi/64 * (Outer.^4 - Inner.^4) .* (1 - cos(pp .* pi/180))
        Ix = (np.pi / 64) * (outer**4 - inner**4) * (1 - np.cos(np.deg2rad(self.pp)))
        
        # 弾性率の適用 (EI.mのif文ロジック)
        E_vec = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1 # MATLAB style index for logic comparison
            if idx == 1 or idx == 11:
                E_vec[i] = self.mat_props[2] # 900
            elif idx == 2:
                E_vec[i] = self.mat_props[4] # 1900
            else:
                E_vec[i] = self.mat_props[3] # 22000
        
        EIx = Ix * E_vec
        total_EI = np.sum(EIx)
        
        # --- 重量計算 (SpurWeight.m) ---
        # 断面積 (mm^2) * 配向補正? (pp/90)
        # MATLAB: rS = pi/4 * (Outer.^2 - Inner.^2) .* pp/90;
        rS = (np.pi / 4) * (outer**2 - inner**2) * (self.pp / 90.0)
        
        # 重量 (kg/m)
        # 密度係数が層によって違う
        weights = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                # MATLAB: weight(1) = 0.001496 * 1000 * rS(1);
                weights[i] = 0.001496 * 1.0 * rS[i] # 1000倍の意味を確認(g->kg?) 今回はそのまま
            else:
                # MATLAB: weight(i) = 0.001559 * 1000 * rS(i);
                weights[i] = 0.001559 * 1.0 * rS[i]
        
        total_weight = np.sum(weights) / 1000.0 # g/m -> kg/m に変換 (係数がg/cc相当と仮定)
        
        return total_EI, total_weight

# ==========================================
# 2. データ生成 (Mass Generation)
# ==========================================

def generate_dataset():
    calc = SparCalculator()
    
    data_EI = []
    data_Weight = []
    data_R = []
    
    # 桁径のバリエーション (mm)
    radii = [40, 50, 60, 70, 80, 90, 100, 110, 120]
    
    print("Generating dataset...")
    
    for R in radii:
        # 積層数の組み合わせ生成
        # 層3～8 (MATLABのi~n相当) はメインの構造部材なのでバリエーションを持たせる
        # 範囲: 1プライ～4プライ程度で振ってみる
        ply_opts_main = [1, 2, 3] 
        # 層1,2,9,10,11などは固定または少ない変動
        ply_opts_fixed = [1]
        
        # 組み合わせ爆発を防ぐため、ランダムサンプリングまたは簡易ループにする
        # ここでは層3,4,5,6の組み合わせを総当たりし、他は固定する簡易モデルでテスト
        
        # ベースの積層構成
        base_plies = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        # 可変部分のループ (層番号2～7: MATLABのindex 3-8)
        # itertools.productを使って全通りの組み合わせを作る
        iter_ranges = [range(0, 5) for _ in range(5)] # 0枚～4枚を5層分
        
        for p_counts in itertools.product(*iter_ranges):
            # プライ数をセット
            current_plies = base_plies.copy()
            # 層2(index 2) から 層6(index 6) までを変動させる
            for k, val in enumerate(p_counts):
                current_plies[k+2] = val + 1 # 1枚～5枚
            
            EI, W = calc.calculate_spec(current_plies, R)
            
            data_EI.append(EI)
            data_Weight.append(W)
            data_R.append(R)

    return np.array(data_EI), np.array(data_Weight), np.array(data_R)

# ==========================================
# 3. 近似と誤差解析 (Analysis)
# ==========================================

def fit_function(x, a, b, c):
    return a * (x ** b) + c

def analyze_and_plot():
    EI_raw, W_raw, R_raw = generate_dataset()
    
    # EIの単位が大きいので、見やすくするためにスケーリングする (任意)
    # ここでは生データのまま扱う
    
    # --- 回帰分析 (Curve Fitting) ---
    # 初期値の推定
    p0 = [0.0001, 0.5, 0.1]
    try:
        popt, pcov = curve_fit(fit_function, EI_raw, W_raw, p0=p0, maxfev=10000)
    except:
        print("Fitting failed. Try adjusting initial parameters or data range.")
        return

    W_pred = fit_function(EI_raw, *popt)
    residuals = W_raw - W_pred
    
    # 決定係数 R^2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((W_raw - np.mean(W_raw))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"Fitting Parameters: A={popt[0]:.4e}, B={popt[1]:.4f}, C={popt[2]:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    # --- プロット作成 ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. EI vs Weight (Scatter & Fit)
    axs[0, 0].scatter(EI_raw, W_raw, alpha=0.3, s=5, label='Actual Designs (Discrete)')
    # ソートしてラインを描画
    sort_idx = np.argsort(EI_raw)
    axs[0, 0].plot(EI_raw[sort_idx], W_pred[sort_idx], color='red', linewidth=2, label='Surrogate Model')
    axs[0, 0].set_xlabel('Bending Stiffness EI [N mm^2]')
    axs[0, 0].set_ylabel('Linear Density [kg/m]')
    axs[0, 0].set_title(f'Stiffness vs Weight Correlation\n(R^2 = {r_squared:.4f})')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-")

    # 2. Residuals (Error) vs Stiffness
    # 剛性が高いところと低いところで誤差の傾向が変わるか？
    axs[0, 1].scatter(EI_raw, residuals, alpha=0.3, s=5, color='green')
    axs[0, 1].axhline(0, color='black', linestyle='--')
    axs[0, 1].set_xlabel('Bending Stiffness EI')
    axs[0, 1].set_ylabel('Residual (Actual - Predicted) [kg/m]')
    axs[0, 1].set_title('Prediction Error vs Stiffness')
    axs[0, 1].grid(True)

    # 3. Error Distribution (Histogram)
    axs[1, 0].hist(residuals, bins=50, color='purple', alpha=0.7)
    axs[1, 0].set_xlabel('Error [kg/m]')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Error Distribution Histogram')
    axs[1, 0].grid(True)
    
    # 4. Relative Error (%)
    rel_error = (residuals / W_raw) * 100
    axs[1, 1].scatter(EI_raw, rel_error, alpha=0.3, s=5, color='orange')
    axs[1, 1].axhline(0, color='black', linestyle='--')
    axs[1, 1].set_xlabel('Bending Stiffness EI')
    axs[1, 1].set_ylabel('Relative Error [%]')
    axs[1, 1].set_title('Relative Error (%) vs Stiffness')
    axs[1, 1].set_ylim(-20, 20) # ±20%の範囲を表示
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()