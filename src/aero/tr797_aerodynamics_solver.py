import numpy as np
import sys
import os

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TR797AerodynamicsSolver:
    """
    R.T. Jonesの理論（TR-797）に基づき、構造制約パラメータ beta から、
    最適な循環分布と誘導抗力を計算するクラス。
    """
    def __init__(self, span_m, num_points=50):
        self.b = span_m
        self.N = num_points
        self.le = span_m / 2.0 # 半スパン
        
        # 揚力線理論のグリッド設定 (翼端から翼根まで)
        self.delta_S = np.ones(self.N) * (self.le / self.N / 2.0)
        self.y = np.linspace(self.delta_S[0], self.le - self.delta_S[-1], self.N)
        
        # 平面翼を仮定 (z=0, fai=0)
        self.z = np.zeros(self.N)
        self.fai = np.zeros(self.N) 
        
        # 事前計算パラメータ
        self.param = self._calc_param()
        self.Q = self._calc_Q()
        self.q, self.norm_vars = self._normalize()

    def _calc_param(self):
        y = self.y[:, np.newaxis]
        z = self.z[:, np.newaxis]
        fai = self.fai[:, np.newaxis]
        y_row = self.y[np.newaxis, :]
        z_row = self.z[np.newaxis, :]
        fai_row = self.fai[np.newaxis, :]
        dS = self.delta_S[np.newaxis, :]
        
        ydash = (y - y_row) * np.cos(fai_row) + (z - z_row) * np.sin(fai_row)
        zdash = -(y - y_row) * np.sin(fai_row) + (z - z_row) * np.cos(fai_row)
        y2dash = (y + y_row) * np.cos(fai_row) + (z - z_row) * np.sin(fai_row)
        z2dash = -(y + y_row) * np.sin(fai_row) + (z - z_row) * np.cos(fai_row)
        
        R_plus = (ydash - dS)**2 + zdash**2
        R_minus = (ydash + dS)**2 + zdash**2
        Rdash_plus = (y2dash + dS)**2 + z2dash**2
        Rdash_minus = (y2dash - dS)**2 + z2dash**2
        
        return {
            'ydash': ydash, 'zdash': zdash,
            'y2dash': y2dash, 'z2dash': z2dash,
            'R_plus': R_plus, 'R_minus': R_minus,
            'Rdash_plus': Rdash_plus, 'Rdash_minus': Rdash_minus
        }

    def _calc_Q(self):
        p = self.param
        dS = self.delta_S[np.newaxis, :]
        fai = self.fai[:, np.newaxis]
        fai_row = self.fai[np.newaxis, :]
        
        t1 = (-(p['ydash'] - dS) / p['R_plus'] + (p['ydash'] + dS) / p['R_minus']) * np.cos(fai - fai_row)
        t2 = (-p['zdash'] / p['R_plus'] + p['zdash'] / p['R_minus']) * np.sin(fai - fai_row)
        t3 = (-(p['y2dash'] - dS) / p['Rdash_minus'] + (p['y2dash'] + dS) / p['Rdash_plus']) * np.cos(fai + fai_row)
        t4 = (-p['z2dash'] / p['Rdash_minus'] + p['z2dash'] / p['Rdash_plus']) * np.sin(fai + fai_row)
        
        Q = (1.0 / (2 * np.pi)) * (t1 + t2 + t3 + t4)
        return Q

    def _normalize(self):
        q = self.Q * self.le
        norm_vars = {
            'delta_sigma': self.delta_S / self.le,
            'eta': self.y / self.le,
            'zeta': self.z / self.le
        }
        return q, norm_vars

    def solve_gamma_and_drag(self, beta, lift_N, rho, V):
        N = self.N
        q_mat = self.q
        dS = self.norm_vars['delta_sigma']
        eta = self.norm_vars['eta']
        
        A = np.pi * q_mat * dS[np.newaxis, :] 
        c_vec = 2 * dS
        b_vec = 3 * np.pi / 2 * dS * eta
        
        TL = A + A.T
        TR = np.column_stack((-c_vec, -b_vec))
        BL = np.vstack((-c_vec, -b_vec))
        BR = np.zeros((2, 2))
        A_dash = np.block([[TL, TR], [BL, BR]])
        
        rhs = np.zeros(N + 2)
        rhs[N] = -1.0
        rhs[N+1] = -beta
        
        try:
            solution = np.linalg.solve(A_dash, rhs)
            g = solution[:N] 
        except np.linalg.LinAlgError:
            return np.full_like(self.y, np.nan), np.nan, np.nan

        gamma_phys = (lift_N / (2 * self.le * rho * V)) * g
        
        eff_inv = g.T @ A @ g
        e = 1.0 / eff_inv
        
        le_sq = self.le**2
        InducedDrag = (lift_N**2 / (2 * np.pi * rho * V**2 * le_sq)) * eff_inv
        
        return gamma_phys, InducedDrag, e