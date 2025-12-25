import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

CSV_PATH = os.path.join("data", "processed", "pareto_spar_dataset_si.csv")
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")

# クラス定義 (必須)
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        if X.ndim == 1: X = X.reshape(1, -1)
        log_ei = X[:, 0]
        r = X[:, 1]
        log_r = np.log10(r + 1e-9)
        thickness_index = log_ei - 3 * log_r
        return np.column_stack((X, thickness_index))

def evaluate_eos_performance():
    print("Evaluating EOS Model (SI Units)...")
    
    df = pd.read_csv(CSV_PATH)
    # 入力もLog(EI_N)を使用
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y_actual = df['Weight'].values

    model = joblib.load(MODEL_PATH)
    
    # 予測
    y_pred_log = model.predict(X)
    y_pred = 10**y_pred_log
    
    # スコア
    r2 = r2_score(y_actual, y_pred)
    print(f"R2 Score: {r2:.5f}")
    
    # プロット
    plt.figure(figsize=(8,8))
    plt.scatter(y_actual, y_pred, alpha=0.3)
    plt.plot([0, max(y_actual)], [0, max(y_actual)], 'r--')
    plt.xlabel("Actual Weight [kg/m]")
    plt.ylabel("Predicted Weight [kg/m]")
    plt.title(f"Evaluation (SI Units) R2={r2:.4f}")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    evaluate_eos_performance()