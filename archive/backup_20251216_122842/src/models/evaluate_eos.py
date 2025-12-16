import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 設定
CSV_PATH = os.path.join("data", "processed", "pareto_spar_dataset_eos_model.csv")
MODEL_PATH = os.path.join("results", "models", "spar_weight_surrogate_model_eos_xgb.pkl")
SAVE_IMG_PATH = os.path.join("results", "figures", "eos_model_evaluation.png")

# =========================================================
# 【重要】学習時と同じ関数をここに定義する必要がある
# =========================================================
def add_physics_features(X):
    """
    物理的なヒントを追加する関数 (学習時と全く同じロジック)
    """
    log_ei = X[:, 0]
    r = X[:, 1]
    
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    """Log10の逆関数: 10^x"""
    return 10**x
# =========================================================


def evaluate_eos_performance():
    # 1. データとモデルの読み込み
    if not os.path.exists(CSV_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Data or Model file not found.")
        return

    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Loading model from {MODEL_PATH}...")
    # joblibは学習時と同じ関数を探すため、上記に関数定義が必要
    model = joblib.load(MODEL_PATH)

    # 2. 前処理 (学習時と同じ処理)
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y_actual = df['Weight'].values

    # 3. テストデータの分割 (学習時の random_state=42 と合わせる)
    _, X_test, _, y_test = train_test_split(X, y_actual, test_size=0.2, random_state=42)
    
    # 4. 予測実行
    y_pred = model.predict(X_test)
    
    # 5. 誤差計算
    # 相対誤差 (%) = (予測 - 実測) / 実測 * 100
    rel_error = (y_pred - y_test) / (y_test + 1e-9) * 100
    
    # 復元 (プロット用)
    log_EI_test = X_test[:, 0]
    R_test = X_test[:, 1]

    # スコア算出
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("-" * 50)
    print(f"Test R^2 Score : {r2:.5f}")
    print(f"Test MAE       : {mae:.5f} kg/m")
    print(f"Mean Rel Error : {np.mean(rel_error):.3f} %")
    print(f"Std Rel Error  : {np.std(rel_error):.3f} %")
    print("-" * 50)

    # 6. プロット作成 (ダッシュボード)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Eos Model Evaluation Dashboard (Log-Target XGBoost)\nR2={r2:.4f}, MAE={mae:.4f} kg/m", fontsize=16)

    # --- 1. Accuracy Check ---
    ax1 = fig.add_subplot(2, 2, 1)
    sc1 = ax1.scatter(y_test, y_pred, c=np.abs(rel_error), cmap='coolwarm', s=15, alpha=0.7, vmin=0, vmax=5) # vmaxを広げる
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.6, label='Ideal')
    ax1.set_xlabel('Actual Weight [kg/m]')
    ax1.set_ylabel('Predicted Weight [kg/m]')
    ax1.set_title('Accuracy Check')
    plt.colorbar(sc1, ax=ax1, label='Abs Relative Error (%)')
    ax1.grid(True, alpha=0.3)

    # --- 2. Error Histogram ---
    ax2 = fig.add_subplot(2, 2, 2)
    mean_err = np.mean(rel_error)
    std_err = np.std(rel_error)
    ax2.hist(rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=1)
    ax2.axvline(mean_err, color='orange', linestyle='-', lw=2, label=f'Mean: {mean_err:.3f}%')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_title(f'Error Distribution\nMean: {mean_err:.3f}%, Std: {std_err:.3f}%')
    ax2.grid(True, alpha=0.3)

    # --- 3. Bias vs Diameter ---
    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(R_test, rel_error, c=log_EI_test, cmap='viridis', s=15, alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--', lw=1)
    ax3.set_xlabel('Diameter R [mm]')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Bias vs Diameter')
    plt.colorbar(sc3, ax=ax3, label='Log10(EI)')
    ax3.grid(True, alpha=0.3)

    # --- 4. Bias vs Stiffness ---
    ax4 = fig.add_subplot(2, 2, 4)
    sc4 = ax4.scatter(log_EI_test, rel_error, c=R_test, cmap='plasma', s=15, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', lw=1)
    ax4.set_xlabel('Log10(EI) [kgf mm^2]')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Bias vs Stiffness')
    plt.colorbar(sc4, ax=ax4, label='Diameter R [mm]')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_IMG_PATH)
    print(f"Evaluation plot saved to: {SAVE_IMG_PATH}")
    plt.show()

if __name__ == "__main__":
    evaluate_eos_performance()