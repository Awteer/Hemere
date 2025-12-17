import numpy as np
from scipy.integrate import simpson

class BendingMomentCalculator:
    """
    揚力分布、自重、幾何形状に基づき、翼のせん断力分布 (V) と
    曲げモーメント分布 (BMD: M) を計算するクラス。
    """

    def __init__(self, span_m, fixed_wing_weight_kg=8.0):
        self.b = span_m
        self.W_fixed_wing = fixed_wing_weight_kg
        self.g = 9.80665
        
    def get_geometry(self, y_coords):
        # 簡易テーパー翼モデル (翼根0.8m -> 翼端0.4m)
        b_half = self.b / 2.0
        eta = np.abs(y_coords) / b_half
        chord = 0.8 * (1 - eta) + 0.4 * eta
        return chord

    def calculate_moment(self, y_aero_m, lift_dist_N_m, spar_weight_dist_kg_m):
        """
        定義式に基づく直接積分法でモーメントを計算する。
        M(y) = Integral_{y}^{tip} [ NetLoad(xi) * (xi - y) ] dxi
        """
        # 1. 荷重分布の計算
        # 固定重量 (N/m)
        w_fixed_N_m = (self.W_fixed_wing / self.b) * self.g
        # スパー重量 (N/m)
        w_spar_N_m = spar_weight_dist_kg_m * self.g
        
        # 正味荷重 (Lift - Weight) [N/m]
        # 上向きを正とする
        net_load_dist = lift_dist_N_m - (w_spar_N_m + w_fixed_N_m)
        
        # 2. モーメントとせん断力の計算
        # 配列サイズ
        N = len(y_aero_m)
        V_dist = np.zeros(N)
        M_dist = np.zeros(N)
        
        # 各スパン位置 y_i について、それより外側(y_j > y_i)の荷重を積分する
        # ※二重ループになるが、点数N=50程度なら計算負荷は無視できるため、
        #   累積誤差のないこの方法が最も確実です。
        
        for i in range(N):
            y_curr = y_aero_m[i]
            
            # 積分区間: 現在位置 i から 翼端(最後) まで
            # y_aero_m は 0 -> b/2 (昇順) と仮定
            if i == N - 1:
                V_dist[i] = 0
                M_dist[i] = 0
                continue
                
            y_outer = y_aero_m[i:]
            load_outer = net_load_dist[i:]
            
            # せん断力 V(y) = Integral(Load)
            # 上向き荷重の積分 -> 上向きのせん断力 (+V)
            V_dist[i] = simpson(load_outer, x=y_outer)
            
            # 曲げモーメント M(y) = Integral( Load * (xi - y) )
            # 腕の長さ (xi - y)
            moment_arm = y_outer - y_curr
            integrand = load_outer * moment_arm
            
            # 上向き荷重 -> プラスの曲げモーメント (と仮定し、後で符号調整)
            # HPAでは「上に反る」モーメントを正と扱うことが多いが、
            # 構造計算(EOS)では絶対値を使うため、符号は一貫していればよい。
            M_dist[i] = simpson(integrand, x=y_outer)
            
        return V_dist, M_dist, net_load_dist

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- 検証用設定 ---
    # 計算精度を確認するため、手計算で解がわかる「等分布荷重」でテストします。
    # 片持ち梁（長さ L）に等分布荷重 w がかかる場合：
    # せん断力 V(y) = w * (L - y)
    # モーメント M(y) = (w / 2) * (L - y)^2
    
    # パラメータ設定
    SPAN_M = 20.0
    HALF_SPAN = SPAN_M / 2.0  # L = 10.0m
    N_POINTS = 101            # 分割数 (Simpson則は奇数点だと尚良い)
    
    # 座標系 (0 -> 10m)
    y_coords = np.linspace(0, HALF_SPAN, N_POINTS)
    
    # 1. インスタンス化 (自重の影響を排除して純粋な積分ロジックを見るため weight=0 に設定)
    calculator = BendingMomentCalculator(span_m=SPAN_M, fixed_wing_weight_kg=0.0)
    
    # 2. テスト入力データの作成
    # ネット荷重を 10 N/m 一定とする
    target_net_load = 10.0
    lift_dist_test = np.full(N_POINTS, target_net_load) # 全域で 10 N/m
    spar_weight_test = np.zeros(N_POINTS)               # スパー重量 0
    
    # 3. 計算実行
    V_num, M_num, net_load = calculator.calculate_moment(
        y_aero_m=y_coords,
        lift_dist_N_m=lift_dist_test,
        spar_weight_dist_kg_m=spar_weight_test
    )
    
    # 4. 理論解 (Analytical Solution) の計算
    # V_theory = w * (L - y)
    V_theory = target_net_load * (HALF_SPAN - y_coords)
    
    # M_theory = 0.5 * w * (L - y)^2
    M_theory = 0.5 * target_net_load * (HALF_SPAN - y_coords)**2
    
    # 5. 結果比較と表示
    print("--- Bending Moment Calculator Verification ---")
    print(f"Test Case: Uniform Load w = {target_net_load} N/m, Length = {HALF_SPAN} m")
    
    # 翼根 (Index 0) での値を比較
    print(f"\n[Root Shear Force (V)]")
    print(f"  Numerical : {V_num[0]:.4f} N")
    print(f"  Theory    : {V_theory[0]:.4f} N")
    print(f"  Error     : {abs(V_num[0] - V_theory[0]):.4e} N")
    
    # 翼根曲げモーメント
    print(f"\n[Root Bending Moment (M)]")
    print(f"  Numerical : {M_num[0]:.4f} Nm")
    print(f"  Theory    : {M_theory[0]:.4f} Nm")
    print(f"  Error     : {abs(M_num[0] - M_theory[0]):.4e} Nm")

    # 翼端 (Index -1) での境界条件チェック
    print(f"\n[Tip Boundary Check]")
    print(f"  Tip V (Expect 0): {V_num[-1]:.4f}")
    print(f"  Tip M (Expect 0): {M_num[-1]:.4f}")

    # --- グラフ描画 (可視化) ---
    plt.figure(figsize=(10, 8))
    
    # 荷重分布
    plt.subplot(3, 1, 1)
    plt.plot(y_coords, net_load, label='Net Load (Input)', color='green', linestyle='--')
    plt.ylabel('Load [N/m]')
    plt.title('Load Distribution')
    plt.grid(True)
    
    # せん断力
    plt.subplot(3, 1, 2)
    plt.plot(y_coords, V_num, label='Numerical V', linewidth=2)
    plt.plot(y_coords, V_theory, label='Theory V', linestyle=':', linewidth=2, color='red')
    plt.ylabel('Shear Force [N]')
    plt.title('Shear Force Distribution (V)')
    plt.legend()
    plt.grid(True)
    
    # 曲げモーメント
    plt.subplot(3, 1, 3)
    plt.plot(y_coords, M_num, label='Numerical M', linewidth=2)
    plt.plot(y_coords, M_theory, label='Theory M', linestyle=':', linewidth=2, color='red')
    plt.ylabel('Bending Moment [Nm]')
    plt.xlabel('Span position y [m]')
    plt.title('Bending Moment Distribution (M)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 判定
    if np.allclose(M_num, M_theory, rtol=1e-2):
        print("\n>>> RESULT: VERIFICATION PASSED (Accuracy within 1%)")
    else:
        print("\n>>> RESULT: VERIFICATION FAILED (Check logic or integration steps)")