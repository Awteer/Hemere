import numpy as np
import joblib
import os
import sys
import argparse
import pandas as pd
import unicodedata

# =========================================================
# EOS Model Loading Helper Functions
# =========================================================
# モデル(Pickle)読み込み時の依存関係解決、および特徴量生成用

def add_physics_features(X):
    """
    学習時と同じ特徴量エンジニアリングを再現する関数。
    入力: [Log10(EI), Diameter] (2次元)
    出力: [Log10(EI), Diameter, Thickness_Index] (3次元)
    """
    # 2次元配列であることを保証
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    log_ei = X[:, 0]
    r_diameter = X[:, 1]
    
    # 対数直径 (ゼロ割防止)
    log_r = np.log10(r_diameter + 1e-9)
    
    # 厚み指数 ( EI / R^3 に相当する対数指標 )
    # これが構造効率を表す重要な特徴量として学習されている
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    """目的変数の逆変換用"""
    return 10**x

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
            print("Hint: Make sure 'add_physics_features' is defined in the script context.")
            sys.exit(1)

    def predict(self, ei_value, diameter_mm, input_unit='N'):
        """
        単一点の予測を行う
        input_unit: 'N' (Nmm^2) or 'kgf' (kgfmm^2)
        """
        # 1. 単位変換処理 (SI -> 重力単位)
        if input_unit == 'N':
            ei_kgf = ei_value / self.g
        else:
            ei_kgf = ei_value

        # 2. 前処理 (Log10)
        safe_ei_kgf = max(ei_kgf, 1.0)
        log_ei = np.log10(safe_ei_kgf)
        
        # 3. 基本特徴量 [Log10(EI_kgf), Diameter_mm]
        # ここで「2列」のデータを作ります
        X_basic = np.array([[log_ei, diameter_mm]])
        
        # 4. 予測実行
        # ★修正: 手動での add_physics_features 呼び出しを削除しました。
        # モデル(Pipeline)の中に特徴量生成処理が含まれているため、
        # ここでは「生の2列」を渡すのが正解です。
        try:
            pred_log_w = self.model.predict(X_basic)[0]
        except ValueError as e:
            print(f"[ERROR] Prediction failed. Input shape: {X_basic.shape}")
            raise e
        
        # 5. 実数に戻す
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
        """対話モード (入力エラー対策済み)"""
        print("\n========================================")
        print("   EOS Surrogate Model Inspector")
        print("========================================")
        print("Type 'q' or 'exit' to quit.")
        
        while True:
            print("\n--------------------------------")
            try:
                # --- 1. 直径入力 ---
                d_str = input("Diameter [mm] > ")
                # 全角→半角変換 & 空白削除
                d_str = unicodedata.normalize('NFKC', d_str).strip()
                
                if d_str.lower() in ['q', 'exit']: break
                if not d_str: continue 

                dia = float(d_str)

                # --- 2. EI入力 ---
                prompt = "Required EI [Nmm^2] (input 'k' to switch to kgf mode) > "
                e_str = input(prompt)
                e_str = unicodedata.normalize('NFKC', e_str).strip()
                
                if e_str.lower() in ['q', 'exit']: break
                if not e_str: continue

                unit = 'N'
                # kで始まる場合はkgfモードへ
                if e_str.lower().startswith('k'):
                    unit = 'kgf'
                    print("  -> Switched to kgf input mode.")
                    e_str = input("Required EI [kgf mm^2] > ")
                    e_str = unicodedata.normalize('NFKC', e_str).strip()
                    if e_str.lower() in ['q', 'exit']: break
                
                # 数値変換
                try:
                    ei = float(e_str)
                except ValueError:
                    print(f"[!] Invalid number: '{e_str}'")
                    continue

                # --- 3. 予測実行 ---
                res = self.predict(ei, dia, unit)

                # 結果表示
                print(f"\n[Result]")
                print(f"  Diameter       : {res['Model_Input_Dia']:.1f} mm")
                print(f"  Input EI       : {res['Input_EI_Origin']:.2e} {res['Unit']}mm^2")
                
                # 内部換算値 (kgf)
                internal_ei = 10**res['Model_Input_LogEI']
                print(f"  (Internal)     : {internal_ei:.2e} kgf mm^2")
                
                print(f"  >> Pred Weight : {res['Output_Weight_kg_m']:.4f} kg/m")
                
            except Exception as e:
                print(f"[!] Unexpected Error: {e}")
                import traceback
                traceback.print_exc()

    def run_sweep(self, dia_mm, ei_min, ei_max, steps=100, output_file="eos_sweep.csv"):
        """スイープモード：CSV出力"""
        print(f"\n--- Running Sweep Analysis ---")
        print(f"Diameter: {dia_mm} mm")
        print(f"EI Range: {ei_min:.1e} ~ {ei_max:.1e} Nmm^2")
        
        ei_values = np.linspace(ei_min, ei_max, steps)
        results = []
        
        for ei in ei_values:
            try:
                res = self.predict(ei, dia_mm, input_unit='N')
                results.append({
                    "EI_Nmm2": res['Input_EI_Origin'],
                    "EI_kgfmm2": 10**res['Model_Input_LogEI'],
                    "Diameter_mm": res['Model_Input_Dia'],
                    "Weight_kg_m": res['Output_Weight_kg_m']
                })
            except Exception as e:
                print(f"[Warning] Failed at EI={ei:.2e}: {e}")
            
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"[Done] Saved sweep data to {output_file}")
            print(df.head())
        else:
            print("[Error] No results generated.")

if __name__ == "__main__":
    # 引数処理
    parser = argparse.ArgumentParser(description="EOS Surrogate Model Inspector")
    # デフォルトパスは環境に合わせて調整してください
    default_model_path = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")
    
    parser.add_argument('--model', type=str, default=default_model_path, help='Path to .pkl model file')
    parser.add_argument('--mode', type=str, default='interactive', choices=['interactive', 'sweep'], help='Operation mode')
    
    # Sweep用引数
    parser.add_argument('--dia', type=float, default=80.0, help='Diameter for sweep [mm]')
    parser.add_argument('--ei_min', type=float, default=1e9, help='Min EI for sweep [Nmm^2]')
    parser.add_argument('--ei_max', type=float, default=1e11, help='Max EI for sweep [Nmm^2]')
    parser.add_argument('--out', type=str, default='eos_validation.csv', help='Output CSV file path')

    args = parser.parse_args()

    # 実行
    inspector = EOSInspector(args.model)
    
    if args.mode == 'interactive':
        inspector.run_interactive()
    elif args.mode == 'sweep':
        inspector.run_sweep(args.dia, args.ei_min, args.ei_max, output_file=args.out)