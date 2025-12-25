import numpy as np
import joblib
import os
import sys
import argparse
import pandas as pd
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

# =========================================================
# EOS Model Loading Helper Class (必須)
# =========================================================
# 学習時と同じクラス定義がないとPickleが復元できません
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # input check
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        log_ei = X[:, 0]
        r = X[:, 1]
        
        # 物理的特徴量: Thickness Index
        log_r = np.log10(r + 1e-9)
        thickness_index = log_ei - 3 * log_r
        
        return np.column_stack((X, thickness_index))

# =========================================================
# EOS Inspector Class
# =========================================================

class EOSInspector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.g = 9.80665
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model file not found: {self.model_path}")
            sys.exit(1)
        try:
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Model loaded successfully: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("Hint: Class 'PhysicsFeatureEngineer' must be defined exactly as in the training script.")
            sys.exit(1)

    def predict(self, ei_value, diameter_mm, input_unit='N'):
        """
        単一点の予測を行う
        input_unit: 'N' (Nmm^2) or 'kgf' (kgfmm^2)
        """
        # 1. 単位変換処理 (SI -> 重力単位ではなく、学習データに合わせる)
        # 今回のRetrainで、学習データ自体を「N単位系」で作った場合は変換不要ですが、
        # もし学習データが「Log10(EI[Nmm^2])」で作られているなら、ここで単純にLogをとるだけです。
        
        # ★重要: 新しい学習データは「Nmm^2」で作った前提で進めます。
        # もしユーザー入力がkgfなら、Nに戻す必要があります。
        
        if input_unit == 'kgf':
            # kgf -> N
            ei_n_mm2 = ei_value * self.g
        else:
            ei_n_mm2 = ei_value

        # 2. 前処理 (Log10)
        # ゼロ割防止
        safe_ei = max(ei_n_mm2, 1.0)
        log_ei = np.log10(safe_ei)
        
        # 3. 基本入力作成 [Log10(EI_N), Diameter_mm]
        X_basic = np.array([[log_ei, diameter_mm]])
        
        # 4. 予測実行
        # パイプライン(PhysicsFeatureEngineer込み)を通すので、生の2列を渡すだけでOK
        try:
            pred_log_w = self.model.predict(X_basic)[0]
        except ValueError as e:
            print(f"[ERROR] Prediction failed. Input shape: {X_basic.shape}")
            raise e
        
        # 5. 実数に戻す (Log10(Weight) -> Weight)
        pred_w_kg_m = 10**pred_log_w
        
        return {
            "Input_EI_Origin": ei_value,
            "Unit": input_unit,
            "Model_Input_LogEI": log_ei,
            "Model_Input_Dia": diameter_mm,
            "Output_LogWeight": pred_log_w,
            "Output_Weight_kg_m": pred_w_kg_m
        }

    def run_interactive(self):
        """対話モード"""
        print("\n========================================")
        print("   EOS Surrogate Model Inspector (New)")
        print("========================================")
        print("Type 'q' or 'exit' to quit.")
        
        while True:
            print("\n--------------------------------")
            try:
                # --- 1. 直径入力 ---
                d_str = input("Diameter [mm] > ")
                d_str = unicodedata.normalize('NFKC', d_str).strip()
                if d_str.lower() in ['q', 'exit']: break
                if not d_str: continue 
                dia = float(d_str)

                # --- 2. EI入力 ---
                prompt = "Required EI [Nmm^2] (input 'k' for kgf mode) > "
                e_str = input(prompt)
                e_str = unicodedata.normalize('NFKC', e_str).strip()
                if e_str.lower() in ['q', 'exit']: break
                if not e_str: continue

                unit = 'N'
                if e_str.lower().startswith('k'):
                    unit = 'kgf'
                    print("  -> Switched to kgf input mode.")
                    e_str = input("Required EI [kgf mm^2] > ")
                    e_str = unicodedata.normalize('NFKC', e_str).strip()
                    if e_str.lower() in ['q', 'exit']: break
                
                ei = float(e_str)

                # --- 3. 予測実行 ---
                res = self.predict(ei, dia, unit)

                # 結果表示
                print(f"\n[Result]")
                print(f"  Diameter       : {res['Model_Input_Dia']:.1f} mm")
                print(f"  Input EI       : {res['Input_EI_Origin']:.2e} {res['Unit']}mm^2")
                # 内部的には Nmm^2 のLogを使っているはず
                print(f"  (Log10 EI)     : {res['Model_Input_LogEI']:.3f}")
                print(f"  >> Pred Weight : {res['Output_Weight_kg_m']:.4f} kg/m")
                
            except Exception as e:
                print(f"[!] Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOS Surrogate Model Inspector")
    default_model_path = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")
    parser.add_argument('--model', type=str, default=default_model_path)
    args = parser.parse_args()

    inspector = EOSInspector(args.model)
    inspector.run_interactive()