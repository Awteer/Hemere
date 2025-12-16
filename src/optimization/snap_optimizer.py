import numpy as np
import pandas as pd
import joblib
import os
import sys
import itertools

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.spar_calculator import SparCalculator

# モデルパス
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# =========================================================
# モデル読み込みに必要な関数定義 (Pickle対策)
# =========================================================
def add_physics_features(X):
    """モデル学習時と同じ特徴量生成"""
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    """Log10の逆変換"""
    return 10**x

# =========================================================
# Snap Optimizer クラス
# =========================================================
class SnapOptimizer:
    def __init__(self):
        # 1. 計算エンジンの初期化
        self.calc = SparCalculator()
        
        # 2. AIモデル(Eos)の読み込み
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Eos Model not found at {MODEL_PATH}. Run eos_surrogate.py first.")
        
        print(f"Loading Eos Engine from {MODEL_PATH}...")
        self.eos_model = joblib.load(MODEL_PATH)
        
        # 探索設定
        self.r_search_step = 0.5  # AI探索時の刻み幅 (mm)
        self.snap_range = 5.0     # AI推奨値の前後何mmを精密探索するか
        self.snap_step = 1.0      # 精密探索時の刻み幅 (mm) -> マンドレル刻みに合わせる

    def predict_ideal_spec(self, target_EI):
        """
        Phase 1: Eosモデルによる理想値(連続値)の探索
        """
        # 30mm ~ 130mm を粗くスキャン
        r_scan = np.arange(30.0, 131.0, self.r_search_step)
        
        # 入力作成 [LogEI, R]
        log_ei = np.log10(target_EI)
        ei_col = np.full_like(r_scan, log_ei)
        X_scan = np.column_stack((ei_col, r_scan))
        
        # 爆速推論
        pred_weights = self.eos_model.predict(X_scan)
        
        # 最適点（最小重量）を見つける
        best_idx = np.argmin(pred_weights)
        ideal_r = r_scan[best_idx]
        ideal_w = pred_weights[best_idx]
        
        return ideal_r, ideal_w

    def snap_to_physics(self, target_EI, ideal_r):
        """
        Phase 2: 物理スナップ
        理想直径の周辺で、実際に製造可能な積層パターンを探索し、
        要求剛性を満たす最軽量解を確定させる。
        """
        # 1. 探索範囲の決定 (AI推奨値 ± snap_range)
        r_min = max(30.0, ideal_r - self.snap_range)
        r_max = min(130.0, ideal_r + self.snap_range)
        
        # マンドレル径は通常1mm刻みなどを想定
        candidate_diameters = np.arange(np.floor(r_min), np.ceil(r_max) + 1.0, self.snap_step)
        
        best_spec = None
        min_weight = float('inf')
        
        # 積層パターンの生成 (ある程度絞り込んで探索)
        # Base層: 0~9, Cap層: 0~2 (全探索に近いが、範囲を絞るロジックを入れても良い)
        # ここでは単純化のため、dataset作成時と同じ全探索ロジックを回す
        # (ただし直径が数点しかないので一瞬で終わる)
        base_options = range(10)
        cap_options = list(itertools.product(range(3), repeat=7))
        
        # --- 精密探索ループ ---
        for D in candidate_diameters:
            # 固定層定義
            ply_counts = np.zeros(11, dtype=int)
            ply_counts[0] = 1; ply_counts[10] = 1; ply_counts[1] = 2

            for base_ply in base_options:
                ply_counts[2] = base_ply
                for cap_config in cap_options:
                    ply_counts[3:10] = cap_config
                    
                    # 物理計算 (SparCalculator)
                    real_EI, real_W, t_total = self.calc.calculate_spec(ply_counts, D)
                    
                    # 判定: 剛性が足りているか？
                    if real_EI >= target_EI:
                        # 判定: 今までのベストより軽いか？
                        if real_W < min_weight:
                            min_weight = real_W
                            best_spec = {
                                "Diameter": D,
                                "Actual_EI": real_EI,
                                "Actual_Weight": real_W,
                                "Thickness": t_total,
                                "Ply_Config": ply_counts.copy().tolist(),
                                "Margin_Pct": (real_EI - target_EI) / target_EI * 100
                            }
                            
        return best_spec

    def solve(self, target_EI):
        """
        メイン処理: AI探索 -> 物理スナップ
        """
        print(f"Target EI: {target_EI:.2e}")
        
        # Step 1: AIによるナビゲーション
        ideal_r, ideal_w = self.predict_ideal_spec(target_EI)
        print(f"  [AI Guide] Ideal Diameter: {ideal_r:.1f} mm (Approx Weight: {ideal_w:.4f} kg/m)")
        
        # Step 2: 物理スナップによる確定
        print(f"  [Physics Snap] Searching feasible specs around {ideal_r:.1f} mm...")
        final_spec = self.snap_to_physics(target_EI, ideal_r)
        
        if final_spec:
            print(f"  ✅ Solution Found!")
            print(f"     Diameter    : {final_spec['Diameter']} mm")
            print(f"     Weight      : {final_spec['Actual_Weight']:.4f} kg/m")
            print(f"     Stiffness   : {final_spec['Actual_EI']:.2e} (Margin: +{final_spec['Margin_Pct']:.1f}%)")
            print(f"     Ply Config  : {final_spec['Ply_Config']}")
            print("-" * 30)
            return final_spec
        else:
            print("  ❌ No feasible solution found in the search range.")
            return None

# =========================================================
# 実行部
# =========================================================
if __name__ == "__main__":
    optimizer = SnapOptimizer()
    
    # テストケース: いくつかの剛性値を試す
    test_targets = [1.0e9, 3.0e9,5.0e9,8.0e9,1.0e10,1.4e10]
    
    for ei in test_targets:
        print("\n" + "="*40)
        optimizer.solve(ei)