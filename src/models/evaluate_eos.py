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
# 【重要】Pickle読み込み用ヘルパー関数
# =========================================================
def add_physics_features(X):
    """
    物理的なヒントを追加する関数 (学習時と全く同じロジック)
    Input: [Log10(EI), R]
    Output: [Log10(EI), R, Thickness_Index]
    """
    # 2次元配列化
    if X.ndim == 1: X = X.reshape(1, -1)
    
    log_ei = X[:, 0]
    r = X[:, 1]
    
    # log(EI) - 3*log(R)
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    return 10**x

# =========================================================

def evaluate_eos_performance():
    print("========================================================")
    print("   EOS Model Rigorous Evaluation")
    print("========================================================")

    # 1. データとモデルの読み込み
    if not os.path.exists(CSV_PATH):
        print(f"[Error] Data file not found: {CSV_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        return

    print(f"Loading data from: {os.path.basename(CSV_PATH)}")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Loading model from: {os.path.basename(MODEL_PATH)}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[Critical Error] Failed to load model. Define helper functions? \n{e}")
        return

    # 2. 前処理 (学習時と同じ処理)
    # 入力: Log10(EI), R (Diameter)
    # 出力(正解): Weight (Linear)
    
    # データセットの単位確認 (重要)
    # csvのEIが Nmm2 なのか kgfmm2 なのか、Weightが kg/m なのかを確認する必要がありますが、
    # ここでは「学習に使ったCSV」そのものを使うため、単位系は整合している前提です。
    
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y_actual_linear = df['Weight'].values # 正解データ（実数）

    # 3. テストデータの分割 (学習時の random_state=42 と合わせる)
    # これにより、学習に使っていない「未知のデータ」に対する性能を測ります
    _, X_test, _, y_test_linear = train_test_split(X, y_actual_linear, test_size=0.2, random_state=42)
    
    print(f"Test Data Size: {len(y_test_linear)} samples")

    # 4. 予測実行
    # モデルは Log10(Weight) を出力するように学習されている可能性が高い
    print("Predicting...")
    y_pred_log = model.predict(X_test)
    
    # ★重要: 対数から実数へ戻す
    y_pred_linear = 10 ** y_pred_log
    
    # 5. 誤差計算
    # 相対誤差 (%) = (予測 - 正解) / 正解 * 100
    rel_error = (y_pred_linear - y_test_linear) / (y_test_linear + 1e-9) * 100
    
    # 復元 (プロット用)
    log_EI_test = X_test[:, 0]
    R_test = X_test[:, 1]

    # スコア算出
    r2 = r2_score(y_test_linear, y_pred_linear)
    mae = mean_absolute_error(y_test_linear, y_pred_linear)
    
    # 統計情報の表示
    print("-" * 50)
    print(f"Test R^2 Score     : {r2:.5f} (1.0 is perfect)")
    print(f"Test MAE           : {mae:.5f} kg/m (Mean Absolute Error)")
    print(f"Mean Rel Error     : {np.mean(rel_error):.3f} %")
    print(f"Std Rel Error      : {np.std(rel_error):.3f} %")
    print(f"Max Rel Error      : {np.max(np.abs(rel_error)):.3f} %")
    print("-" * 50)
    
    # データの中身をチラ見せ (デバッグ用)
    print("Sample Comparison (First 5 test data):")
    print(f"{'Actual [kg/m]':<15} | {'Pred [kg/m]':<15} | {'Diff [%]':<10}")
    for i in range(5):
        print(f"{y_test_linear[i]:<15.4f} | {y_pred_linear[i]:<15.4f} | {rel_error[i]:<10.2f}")
    print("-" * 50)

    # 6. プロット作成 (ダッシュボード)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"EOS Model Evaluation Dashboard\nR2={r2:.4f}, MAE={mae:.4f} kg/m", fontsize=16)

    # --- 1. Accuracy Check (正解 vs 予測) ---
    ax1 = fig.add_subplot(2, 2, 1)
    # 色分けは「誤差の大きさ」で行う
    sc1 = ax1.scatter(y_test_linear, y_pred_linear, c=np.abs(rel_error), cmap='coolwarm', s=15, alpha=0.7, vmin=0, vmax=5)
    
    # 理想線 (y=x)
    min_val = min(y_test_linear.min(), y_pred_linear.min())
    max_val = max(y_test_linear.max(), y_pred_linear.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.6, label='Ideal')
    
    ax1.set_xlabel('Actual Weight (Physics Calculation) [kg/m]')
    ax1.set_ylabel('AI Predicted Weight [kg/m]')
    ax1.set_title('Accuracy Check (Linear Scale)')
    plt.colorbar(sc1, ax=ax1, label='Abs Relative Error (%)')
    ax1.grid(True, alpha=0.3)

    # --- 2. Error Histogram (誤差の分布) ---
    ax2 = fig.add_subplot(2, 2, 2)
    mean_err = np.mean(rel_error)
    std_err = np.std(rel_error)
    ax2.hist(rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=1)
    ax2.axvline(mean_err, color='orange', linestyle='-', lw=2, label=f'Mean: {mean_err:.3f}%')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_title(f'Error Distribution\nMean: {mean_err:.3f}%, Std: {std_err:.3f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. Bias vs Diameter (直径による誤差の偏り) ---
    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(R_test, rel_error, c=log_EI_test, cmap='viridis', s=15, alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--', lw=1)
    ax3.set_xlabel('Diameter R [mm]')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Bias vs Diameter')
    plt.colorbar(sc3, ax=ax3, label='Log10(EI)')
    ax3.grid(True, alpha=0.3)

    # --- 4. Bias vs Stiffness (剛性による誤差の偏り) ---
    ax4 = fig.add_subplot(2, 2, 4)
    sc4 = ax4.scatter(log_EI_test, rel_error, c=R_test, cmap='plasma', s=15, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', lw=1)
    ax4.set_xlabel('Log10(EI) [kgf mm^2]')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Bias vs Stiffness')
    plt.colorbar(sc4, ax=ax4, label='Diameter R [mm]')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 保存ディレクトリ作成
    os.makedirs(os.path.dirname(SAVE_IMG_PATH), exist_ok=True)
    plt.savefig(SAVE_IMG_PATH)
    print(f"Evaluation plot saved to: {SAVE_IMG_PATH}")
    plt.show()

if __name__ == "__main__":
    evaluate_eos_performance()