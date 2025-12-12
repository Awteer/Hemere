import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# ==========================================
# 1. サロゲートモデル (The "Magic" Formula)
# ==========================================
def get_spar_linear_density(EI_Nmm2, diameter_mm, safety_factor=1.03):
    """
    曲げ剛性 EI [N*mm^2] と 直径 D [mm] から
    単位長さあたりの重量 [kg/m] を予測する
    """
    # Elite 10% parameters (N=1,000,000)
    A = 1.890215e-09
    B = 0.826560
    C = -1.296560
    D = 0.000059
    
    # モデル式 (EIは正の値であること)
    # EIが極端に小さいとエラーになるのを防ぐ
    EI_safe = np.maximum(EI_Nmm2, 1.0)
    raw_weight = A * (EI_safe ** B) * (diameter_mm ** C) + D
    
    return raw_weight * safety_factor

# ==========================================
# 2. 空力モデル (Ported from opt_ATLAS / TR-797)
# ==========================================
class AtlasAerodynamics:
    def __init__(self, span_m, V_flight, rho_air, N_segments=40):
        self.b = span_m
        self.le = span_m / 2.0
        self.Uinf = V_flight
        self.rho = rho_air
        self.N = N_segments
        
        # Geometry discretization (Cosine spacing or Linear?)
        # opt_ATLAS uses linear spacing for delta_S but defined y linearly?
        # Let's follow opt_ATLAS: delta_S is constant (equal distance)
        self.delta_S = np.ones(self.N) * (self.le / self.N / 2.0)
        self.y = np.linspace(self.delta_S[0], self.le - self.delta_S[-1], self.N)
        self.z = np.zeros(self.N)
        self.phi = np.zeros(self.N) # Dihedral/Twist geometry angle (0 for planar)
        
        # Precompute Aerodynamic Matrices (Q, A)
        self._precompute_matrices()

    def _precompute_matrices(self):
        # Python broadcasting to replace nested loops
        y, z, phi = self.y, self.z, self.phi
        y_col = y[:, np.newaxis]
        z_col = z[:, np.newaxis]
        phi_col = phi[:, np.newaxis]
        
        # Relative coordinates
        ydash = (y_col - y) * np.cos(phi) + (z_col - z) * np.sin(phi)
        zdash = -(y_col - y) * np.sin(phi) + (z_col - z) * np.cos(phi)
        y2dash = (y_col + y) * np.cos(phi) + (z_col - z) * np.sin(phi)
        z2dash = -(y_col + y) * np.sin(phi) + (z_col - z) * np.cos(phi)
        
        dS = self.delta_S
        R_plus = (ydash - dS)**2 + zdash**2
        R_minus = (ydash + dS)**2 + zdash**2
        Rdash_plus = (y2dash + dS)**2 + z2dash**2
        Rdash_minus = (y2dash - dS)**2 + z2dash**2
        
        # Q matrix calculation
        # Avoid division by zero on diagonal (self-induction handled implicitly or skipped?)
        # In TR-797 code, diagonal terms might need care, but let's implement direct formula
        # Assuming points are control points and don't overlap with vortex edges
        
        term1 = (-(ydash - dS)/R_plus + (ydash + dS)/R_minus) * np.cos(phi_col - phi)
        term2 = (-zdash/R_plus + zdash/R_minus) * np.sin(phi_col - phi)
        term3 = (-(y2dash - dS)/Rdash_minus + (y2dash + dS)/Rdash_plus) * np.cos(phi_col + phi)
        term4 = (-z2dash/Rdash_minus + z2dash/Rdash_plus) * np.sin(phi_col + phi)
        
        self.Q = (1 / (2 * np.pi)) * (term1 + term2 + term3 + term4)
        
        # Normalize to q and A
        self.q = self.Q * self.le
        self.delta_sigma = self.delta_S / self.le
        self.eta = self.y / self.le
        
        # Matrix A for linear system
        # A_ij = pi * q_ij * delta_sigma_i
        self.A = np.pi * self.q * self.delta_sigma[:, np.newaxis]

    def solve_circulation(self, beta, Lift):
        """
        Solves for circulation distribution Gamma given beta and Total Lift
        """
        N = self.N
        c_vec = 2 * self.delta_sigma
        b_vec = 3 * np.pi / 2 * self.delta_sigma * self.eta
        
        # Construct Linear System: [A+A', -c, -b; ...] * [g; mu1; mu2] = [0; -1; -beta]
        # Top-left block
        TL = self.A + self.A.T
        
        # Borders
        # row: -c^T, -b^T
        # col: -c, -b
        
        # Big Matrix construction
        # Size (N+2) x (N+2)
        SysMat = np.zeros((N + 2, N + 2))
        SysMat[:N, :N] = TL
        SysMat[:N, N] = -c_vec
        SysMat[:N, N+1] = -b_vec
        SysMat[N, :N] = -c_vec
        SysMat[N+1, :N] = -b_vec
        
        # Right Hand Side
        RHS = np.zeros(N + 2)
        RHS[N] = -1.0
        RHS[N+1] = -beta
        
        # Solve
        try:
            solution = np.linalg.solve(SysMat, RHS)
            g = solution[:N]
        except np.linalg.LinAlgError:
            g = np.full(N, np.nan)
            
        # Calculate Induced Drag Efficiency
        # efficiency_inverse = g' * A * g
        efficiency_inv = g.T @ self.A @ g
        efficiency = 1.0 / efficiency_inv
        
        # Dimensional Lift and Drag
        # Gamma = (L / (2 * le * rho * U)) * g
        Gamma_dim = (Lift / (2 * self.le * self.rho * self.Uinf)) * g
        
        # Induced Drag
        # Di_ellipse = L^2 / (2*pi*rho*U^2 * le^2)
        Di_ellipse = Lift**2 / (2 * np.pi * self.rho * self.Uinf**2 * self.le**2)
        Di = efficiency_inv * Di_ellipse
        
        # Lift distribution (N/m) = rho * U * Gamma
        L_dist = self.rho * self.Uinf * Gamma_dim
        
        return g, Gamma_dim, L_dist, Di, efficiency

# ==========================================
# 3. 構造 & 統合評価関数
# ==========================================
def evaluate_design(x, aero_model, params, detailed=False):
    """
    x: Design Variables
       x[0]: beta (Circulation parameter)
       x[1:]: EI distribution (log10 scale)
    """
    beta = x[0]
    log_EI = x[1:]
    EI_dist_Nmm2 = 10**log_EI
    
    # 1. 重量推算 (Surrogate Model)
    # 桁径分布 (固定値またはパラメータから取得)
    # ここでは opt_ATLAS のように根元から翼端へテーパーさせると仮定
    R_dist_mm = np.linspace(params['R_root'], params['R_tip'], aero_model.N)
    
    # サロゲートモデルで重量密度計算
    w_spar_kg_m = get_spar_linear_density(EI_dist_Nmm2, R_dist_mm)
    
    # 翼重量 (桁 + 二次構造)
    # 積分: sum(w * dy) * 2 (両翼)
    dy = aero_model.y[1] - aero_model.y[0] # Assume const spacing
    W_spar_kg = np.sum(w_spar_kg_m) * dy * 2
    W_fixed_kg = params['W_fixed'] # リブや外皮など
    W_total_kg = params['W_pilot'] + W_spar_kg + W_fixed_kg
    W_total_N = W_total_kg * 9.81
    
    # 2. 空力計算 (揚力 = 重量 となるように循環をスケーリング)
    # TR-797ソルバーは正規化されているので、Liftを入力すればGammaが求まる
    g, Gamma, L_dist, Di, eff = aero_model.solve_circulation(beta, W_total_N)
    
    # 3. 構造解析 (たわみ・応力)
    # 正味荷重 (Lift - Weight)
    # 構造重量分布 [N/m]
    w_struct_N_m = w_spar_kg_m * 9.81 + (W_fixed_kg * 9.81 / params['span'])
    net_load = L_dist - w_struct_N_m
    
    # 曲げモーメント (翼端から積分)
    M_dist = np.zeros_like(net_load)
    shear = 0
    moment = 0
    # Reverse loop for integration from tip
    for i in range(aero_model.N - 1, -1, -1):
        shear += net_load[i] * dy
        moment += shear * dy # Simplified Euler method
        M_dist[i] = moment
        
    # たわみ (翼根から積分)
    # curvature = M / EI
    # EI is in N*mm^2 -> Convert to N*m^2 for integration? 
    # M is N*m (if dy in m). EI input N*mm^2 = N*m^2 * 1e-6?
    # No, N*mm^2 = N * (1e-3 m)^2 = 1e-6 N*m^2.
    EI_Nm2 = EI_dist_Nmm2 * 1e-6
    
    curvature = M_dist / EI_Nm2
    theta = 0
    deflection = 0
    delta_tip = 0
    # Forward loop
    for i in range(aero_model.N):
        theta += curvature[i] * dy
        deflection += theta * dy
        
    delta_tip = deflection # Tip deflection [m]
    
    # 応力 (簡易計算: sigma = M * (D/2) / I)
    # I approx = EI / E_modulus? No, we have EI.
    # Need to assume specific modulus or just limit Strain?
    # Or optimize EI directly. Let's use Strain constraint or approximate Stress.
    # Assuming thin walled pipe: sigma approx = M * R / I
    # EI = E * I -> I = EI / E. 
    # sigma = M * R / (EI / E) = M * R * E / EI.
    # Assume E_effective ~ 100 GPa (CFRP)
    E_eff = 100e9 # Pa
    sigma = (M_dist * (R_dist_mm/1000/2) * E_eff) / EI_Nm2 # Pa
    max_stress = np.max(sigma)

    # 4. 目的関数とペナルティ
    # Objective: Minimize Di
    obj = Di
    
    # Constraints (Penalties for simple scalar minimization)
    # Deflection Constraint
    penalty = 0
    if delta_tip > params['max_deflection']:
        penalty += (delta_tip - params['max_deflection'])**2 * 1e5
        
    # Stress Constraint
    if max_stress > params['max_stress']:
        penalty += (max_stress - params['max_stress'])**2 * 1e-15 # Scale factor needed
        
    total_cost = obj + penalty
    
    if detailed:
        return {
            'beta': beta,
            'W_total_kg': W_total_kg,
            'Di': Di,
            'delta_tip_m': delta_tip,
            'max_stress_MPa': max_stress / 1e6,
            'EI_dist': EI_dist_Nmm2,
            'Lift_dist': L_dist,
            'Moment_dist': M_dist
        }
    else:
        return total_cost

# ==========================================
# 4. メイン実行部
# ==========================================
def run_verification():
    # パラメータ設定
    span = 30.0 # m
    params = {
        'span': span,
        'W_pilot': 60.0, # kg
        'W_fixed': 20.0, # kg (Ribs, film, etc)
        'R_root': 120.0, # mm (Diameter)
        'R_tip': 50.0,   # mm
        'max_deflection': 2.2, # m
        'max_stress': 300e6, # Pa
    }
    
    # 空力モデル初期化
    aero = AtlasAerodynamics(span_m=span, V_flight=7.5, rho_air=1.154, N_segments=20)
    
    # 初期値
    # x = [beta, log10(EI_0), ..., log10(EI_N-1)]
    beta_init = 0.9 # Beta < 1 reduces root moment
    EI_init = np.linspace(np.log10(2e10), np.log10(1e9), 20)
    x0 = np.concatenate(([beta_init], EI_init))
    
    print("Optimization Start...")
    print(f"Initial Beta: {beta_init}")
    
    # 最適化実行 (SLSQPだと制約を別定義できるが、今回はペナルティ法で簡易実装)
    # 変数が多いので BFGS や Nelder-Mead は遅いかも。SLSQP推奨だが...
    # ここでは動作確認のため Nelder-Mead で軽く回すか、勾配法を使う
    
    # Bounds
    bounds = [(0.5, 1.5)] + [(8.0, 11.0)] * 20 # Beta: 0.5-1.5, EI: 10^8 - 10^11
    
    res = minimize(
        evaluate_design, 
        x0, 
        args=(aero, params), 
        method='L-BFGS-B', # Bounds supported
        bounds=bounds,
        options={'disp': True, 'maxiter': 500}
    )
    
    print("Optimization Finished.")
    
    # 結果詳細
    best_design = evaluate_design(res.x, aero, params, detailed=True)
    
    print("-" * 40)
    print(f"Optimal Beta: {best_design['beta']:.4f}")
    print(f"Total Weight: {best_design['W_total_kg']:.2f} kg")
    print(f"Induced Drag: {best_design['Di']:.4f} N")
    print(f"Tip Deflect : {best_design['delta_tip_m']:.3f} m")
    print(f"Max Stress  : {best_design['max_stress_MPa']:.1f} MPa")
    print("-" * 40)
    
    # プロット
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    y = aero.y
    
    # 1. Lift Distribution
    ax[0, 0].plot(y, best_design['Lift_dist'], label='Lift')
    ax[0, 0].set_title('Lift Distribution')
    ax[0, 0].set_xlabel('Span [m]')
    ax[0, 0].set_ylabel('Lift [N/m]')
    ax[0, 0].grid()
    
    # 2. EI Distribution
    ax[0, 1].semilogy(y, best_design['EI_dist'], 'r-o', label='Optimized EI')
    ax[0, 1].set_title('Bending Stiffness (EI)')
    ax[0, 1].set_xlabel('Span [m]')
    ax[0, 1].set_ylabel('EI [N mm^2]')
    ax[0, 1].grid()
    
    # 3. Bending Moment
    ax[1, 0].plot(y, best_design['Moment_dist'], 'g-', label='Moment')
    ax[1, 0].set_title('Bending Moment')
    ax[1, 0].grid()
    
    # 4. Weight Distribution (Spar)
    # Recalculate for plot
    R_dist = np.linspace(params['R_root'], params['R_tip'], 20)
    w_spar = get_spar_linear_density(best_design['EI_dist'], R_dist)
    ax[1, 1].plot(y, w_spar, 'k--', label='Spar Weight')
    ax[1, 1].set_title('Spar Weight Distribution [kg/m]')
    ax[1, 1].grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_verification()