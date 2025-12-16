import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# XGBoost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 設定
CSV_PATH = os.path.join("data", "processed", "pareto_spar_dataset_eos_model.csv")
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# ==========================================
# 特徴量エンジニアリング関数
# ==========================================
def add_physics_features(X):
    """
    物理的なヒントを追加する関数。Input X: [Log10(EI), R]
    Log(EI) - 3*Log(R) (肉厚指数) を追加
    """
    log_ei = X[:, 0]
    r = X[:, 1]
    
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

# ==========================================
# ターゲット逆変換関数 (Pickle対応のため通常関数として定義)
# ==========================================
def inverse_log10(x):
    """Log10の逆関数: 10^x"""
    return 10**x

# ==========================================
# 1. データ読み込み
# ==========================================
def load_and_preprocess(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 入力: [LogEI, R]
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    # 出力: Weight (ここではまだ生のまま)
    y = df['Weight'].values
    
    print(f"Data loaded. Samples: {len(df)}")
    return X, y, df

# ==========================================
# 2. 学習 (Log-Target Strategy + XGBoost)
# ==========================================
def train_eos_model_xgb():
    # フォルダ作成
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    X, y, df = load_and_preprocess(CSV_PATH)
    
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ベースとなるXGBoostモデル
    xgb_base = xgb.XGBRegressor(
        n_estimators=1500,     # 木の数
        learning_rate=0.03,    # 学習率 (低く設定)
        max_depth=8,           # 木の深さ (複雑なパレート境界に追従させるため深く)
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )

    # パイプライン (特徴量生成 -> スケーリング -> XGBoost)
    model_pipeline = Pipeline([
        ('feat_eng', FunctionTransformer(add_physics_features)),
        ('scaler', StandardScaler()),
        ('xgb', xgb_base)
    ])

    # ターゲット変換ラッパー (Log(W)を予測させ、不均一分散の問題を解決)
    full_model = TransformedTargetRegressor(
        regressor=model_pipeline,
        func=np.log10,       # 学習前に適用する関数 (Log変換)
        inverse_func=inverse_log10  # 予測後に戻す関数 (10^x)
    )

    print("-" * 50)
    print("Training Eos Model (Log-Space Target Strategy)...")
    full_model.fit(X_train, y_train)
    
    # 評価
    print("Evaluating...")
    train_score = full_model.score(X_train, y_train)
    test_score = full_model.score(X_test, y_test)
    
    y_pred = full_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 相対誤差
    rel_error = (y_pred - y_test) / (y_test + 1e-9) * 100
    std_err = np.std(rel_error)

    print("-" * 50)
    print(f"Training R^2 : {train_score:.5f}")
    print(f"Test R^2     : {test_score:.5f}")
    print(f"Test MAE     : {mae:.5f} kg/m")
    print(f"Error Std    : {std_err:.3f} % (目標: 1.5%以下)")
    print("-" * 50)

    # 保存
    joblib.dump(full_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    return full_model

if __name__ == "__main__":
    train_eos_model_xgb()