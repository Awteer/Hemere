import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# パス設定 (SI版のデータを使う)
DATA_PATH = os.path.join("data", "processed", "pareto_spar_dataset_si.csv")
MODEL_SAVE_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# =========================================================
# 特徴量エンジニアリングクラス (ここが重要)
# =========================================================
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 入力は [Log10(EI_N), Diameter_mm]
        log_ei = X[:, 0]
        r = X[:, 1]
        
        # 物理的特徴量: Thickness Index
        # t ~ EI / r^3 に相関するため、対数空間での線形結合を作る
        log_r = np.log10(r + 1e-9)
        thickness_index = log_ei - 3 * log_r
        
        return np.column_stack((X, thickness_index))

def train_model():
    print("Training EOS Model on SI Units...")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 入力: EI (Nmm^2), R (mm) -> Log10(EI) をとる
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y = df['Weight'].values
    
    # ターゲット: Weightも対数分布に近いのでLogをとって学習
    y_log = np.log10(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # パイプライン構築
    pipeline = Pipeline([
        ('engineer', PhysicsFeatureEngineer()), # カスタムクラス
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train_log)
    
    # 評価
    y_pred_log = pipeline.predict(X_test)
    y_pred = 10**y_pred_log
    y_test = 10**y_test_log
    
    r2 = r2_score(y_test, y_pred)
    print(f"Test R2 Score : {r2:.5f}")
    
    # 保存
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()