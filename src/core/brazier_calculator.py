import numpy as np
import math

class BrazierCalculator:
    """
    「部分的な積層を含めたBrazierの計算法」PDFの第3章（p38-39）に基づき、
    積層角度が階段状に変化する桁のBrazier座屈モーメントを数値積分で計算するクラス。
    """

    def __init__(self, knockdown_factor=0.5):
        """
        Args:
            knockdown_factor (float): 軽減係数 C (デフォルト0.5) [cite: 854]
        """
        self.C = knockdown_factor

    def get_stiffness_at_theta(self, theta_deg, diameter, ply_counts, ply_angles, mat_props_list):
        """
        ある角度 theta における A11, D22 を計算する。
        """
        # 角度の正規化 (0~90度へのマッピング)
        # 上下対称かつ左右対称と仮定し、第1象限だけで計算して4倍...
        # ではなく、PDFの式は全周積分を前提としているため、
        # 各層が存在するかどうかを判定して A11, D22 を積算する。

        # 判定用角度: 上(90度)または下(270度)からの偏差
        # 例: ply_angle=50度なら、90±25度の範囲に存在
        dist_from_top = abs(theta_deg - 90)
        dist_from_bottom = abs(theta_deg - 270)
        
        # 現在のthetaにおいて有効な層をリストアップ
        current_schedule = []
        
        # 各層についてループ (ply_countsは [Glass, Torque, Base, Cap1...Cap7, Glass])
        # ply_anglesは [90, 90, 90, 50, 45, ..., 20, 90] (幅)
        
        current_z = 0.0 # 積層厚み計算用
        
        # 積層順序に従いリスト化 (SparCalculatorの順序に準拠)
        # 1. Glass In (Index 0) - 全周
        if ply_counts[0] > 0:
            current_schedule.append(mat_props_list[0]) # 24t_90

        # 2. Torque (Index 1) - 全周
        if ply_counts[1] > 0:
            for _ in range(ply_counts[1]):
                current_schedule.append(mat_props_list[1]) # 40t_45 (簡易)

        # 3. Base (Index 2) - 全周
        if ply_counts[2] > 0:
            for _ in range(ply_counts[2]):
                current_schedule.append(mat_props_list[2]) # 40t_0

        # 4. Caps (Index 3~9) - 角度制限あり
        # ply_angles[i] は「層の幅(degree)」
        for i in range(3, 10):
            if ply_counts[i] > 0:
                width = ply_angles[i]
                half_width = width / 2.0
                # フランジ領域(上or下)に入っているか？
                in_top = (dist_from_top <= half_width)
                in_bottom = (dist_from_bottom <= half_width)
                
                if in_top or in_bottom:
                    for _ in range(ply_counts[i]):
                        current_schedule.append(mat_props_list[3]) # 40t_0 (Cap)

        # 5. Glass Out (Index 10) - 全周
        if ply_counts[10] > 0:
            current_schedule.append(mat_props_list[0]) # 24t_90
            
        # --- A11, D22 計算 ---
        # 簡易CLT (Cross Lamination Theory)
        A11 = 0.0
        D22 = 0.0
        
        # 全厚計算して中立軸を決定
        total_t = sum(p['thickness'] for p in current_schedule)
        z_curr = -total_t / 2.0
        
        for ply in current_schedule:
            t = ply['thickness']
            Ex = ply['Ex']
            Ey = ply['Ey']
            angle_rad = np.deg2rad(ply['angle'])
            c = np.cos(angle_rad)
            s = np.sin(angle_rad)
            
            # 座標変換 (簡易)
            # A11 (軸方向): Exへの寄与
            Ex_eff = Ex * (c**4) + Ey * (s**4) # + 2*Gxy...省略(近似)
            A11 += Ex_eff * t
            
            # D22 (周方向曲げ): Eyへの寄与
            Ey_eff = Ex * (s**4) + Ey * (c**4)
            z_next = z_curr + t
            D22 += Ey_eff * (z_next**3 - z_curr**3) / 3.0
            
            z_curr = z_next
            
        return A11, D22

    def calculate_max_moment_integral(self, diameter, ply_counts, ply_angles):
        """
        PDF P.39 の数値積分法を用いて座屈モーメントを計算
        """
        r = diameter / 2.0
        
        # 物性マスタ (MPa単位)
        # SparCalculatorの mat_props_kgf と同じ並び順で定義
        # [24t_90, 40t_45, 40t_0, 40t_0(Cap)]
        G = 9.80665
        t40 = 0.111
        t24 = 0.125
        
        # {thickness, Ex, Ey, angle}
        # Note: ここでのExは繊維方向、Eyは直交
        props = [
            {'thickness': t24, 'Ex': 900*G,   'Ey': 13000*G, 'angle': 90}, # Index 0: 24t_90 (Glass) - Exが90度方向
            {'thickness': t40, 'Ex': 22000*G, 'Ey': 800*G,   'angle': 45}, # Index 1: 40t_45 (Torque)
            {'thickness': t40, 'Ex': 22000*G, 'Ey': 800*G,   'angle': 0},  # Index 2: 40t_0  (Base)
            {'thickness': t40, 'Ex': 22000*G, 'Ey': 800*G,   'angle': 0}   # Index 3: 40t_0  (Cap)
        ]
        
        # 積分パラメータの累積和
        Sum_A = 0.0
        Sum_B = 0.0
        Sum_C = 0.0
        
        # 数値積分 (0度～359度)
        # 対称性を利用して 0~90度 だけ計算して4倍することも可能だが、
        # コードの単純化のため全周回す (計算負荷は軽微)
        steps = 360
        d_theta = 2 * np.pi / steps # rad
        
        for deg in range(steps):
            theta_rad = np.deg2rad(deg)
            
            # その角度での剛性を取得
            A11, D22 = self.get_stiffness_at_theta(deg, diameter, ply_counts, ply_angles, props)
            
            # 積分項の計算 [cite: 905, 906, 907]
            # A項: 1/2 * A11 * r^3 * 4*sin^4(theta)
            term_A = 0.5 * A11 * (r**3) * 4 * (np.sin(theta_rad)**4)
            
            # B項: 1/2 * A11 * r^3 * sin^2(theta)
            term_B = 0.5 * A11 * (r**3) * (np.sin(theta_rad)**2)
            
            # C項: 1/2 * (D22/r) * cos^2(2*theta)
            term_C = 0.5 * (D22 / r) * (np.cos(2 * theta_rad)**2)
            
            Sum_A += term_A * d_theta
            Sum_B += term_B * d_theta
            Sum_C += term_C * d_theta
            
        # 最大モーメントの算出 
        # M_max = (4*sqrt(6)/9) * (B^1.5 * C^0.5 / A)
        
        if Sum_A == 0: return 0.0
        
        coeff = (4 * np.sqrt(6)) / 9.0
        M_theoretical = coeff * ( (Sum_B**1.5) * (Sum_C**0.5) ) / Sum_A
        
        # 軽減係数
        M_actual = M_theoretical * self.C
        
        return M_actual