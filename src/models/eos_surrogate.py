import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# XGBoostのインポート
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 設定
CSV_PATH = os.path.join("data", "processed", "pareto_spar_dataset_eos_model.csv")
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl") # ファイル名変更

# ==========================================
# 特徴量エンジニアリング関数
# ==========================================
def add_physics_features(X):
    """
    物理的なヒントを追加する関数
    Input X: [Log10(EI), R]
    Output:  [Log10(EI), R, Log10(EI)/R^3, Log10(EI)/R^4]
    """
    # X[:, 0] -> Log10(EI)
    # X[:, 1] -> R
    
    log_ei = X[:, 0]
    r = X[:, 1]
    
    # ヒント1: 剛性/R^3 (肉厚に比例する指標)
    # 対数空間なので log(EI) - 3*log(R) になるが、割り算の形でも効果はある
    # ここではシンプルに元のスケールに戻して割る、あるいは対数のまま線形結合を作る手助けをする
    
    # 対数のままの幾何学的関係性
    # I ∝ R^3 * t  =>  LogI ∝ 3LogR + Logt
    # AIがこの線形関係を見つけやすくするために、LogRも渡すと良いかも
    
    log_r = np.log10(r + 1e-9)
    
    # 肉厚インデックスのようなもの (LogEI - 3*LogR)
    thickness_index = log_ei - 3 * log_r
    
    # 既存の列に結合
    return np.column_stack((X, thickness_index))

# ==========================================
# 1. データ読み込み
# ==========================================
def load_and_preprocess(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Log10(EI) と R を基本特徴量とする
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y = df['Weight'].values
    
    return X, y, df

# ==========================================
# 2. 学習 (XGBoost)
# ==========================================
def train_eos_model_xgb():
    X, y, df = load_and_preprocess(CSV_PATH)
    
    # 特徴量エンジニアリング適用（分割前に確認）
    # Pipeline内でやるのでここでは分割だけ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoostモデル定義
    # パラメータは少し強めに調整
    model = Pipeline([
        # 物理特徴量の追加
        ('feat_eng', FunctionTransformer(add_physics_features)),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            n_estimators=1000,     # 木の数
            learning_rate=0.05,    # 学習率 (低いほうが精度が出るが遅い)
            max_depth=6,           # 木の深さ (深すぎると過学習)
            subsample=0.8,         # データの一部を使って学習 (過学習防止)
            colsample_bytree=0.8,  # 特徴量の一部を使って学習
            n_jobs=-1,
            random_state=42,
            objective='reg:squarederror'
        ))
    ])

    print("-" * 50)
    print("Training Eos Model (XGBoost + Physics Features)...")
    model.fit(X_train, y_train)
    
    # 評価
    print("Evaluating...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 誤差の統計
    rel_error = (y_pred - y_test) / (y_test + 1e-9) * 100
    std_err = np.std(rel_error)

    print("-" * 50)
    print(f"Training R^2 : {train_score:.5f}")
    print(f"Test R^2     : {test_score:.5f}")
    print(f"Test MAE     : {mae:.5f} kg/m")
    print(f"Error Std    : {std_err:.3f} % (Target: Low variance)")
    print("-" * 50)

    # 保存
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    train_eos_model_xgb()