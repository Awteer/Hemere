import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from mpl_toolkits.mplot3d import Axes3D

# AI / Scikit-Learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ==========================================
# 1. データ読み込み & 前処理
# ==========================================
def load_and_preprocess(csv_path):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 欠損値ケア（念のため）
    df = df.dropna()
    
    # 入力 (X): EI, Radius
    # 出力 (y): Weight
    
    # 【重要】物理的性質に合わせた変換
    # EIは 10^9 ~ 10^12 のオーダーなので、対数変換しないとAIが理解できない
    # RはそのままでOK
    
    # 特徴量行列を作成
    # column 0: Log10(EI)
    # column 1: R (Diameter)
    X = np.column_stack((
        np.log10(df['EI'].values), 
        df['R'].values
    ))
    
    y = df['Weight'].values
    
    print(f"Data loaded. Samples: {len(df)}")
    return X, y, df # 描画用にdfも返す

# ==========================================
# 2. 検証用ダッシュボード (Visualizer)
# ==========================================
class ModelEvaluator:
    def __init__(self, X_val, y_val, model):
        self.X = X_val
        self.y_actual = y_val
        self.y_pred = model.predict(X_val)
        
        # 復元した物理量（プロット用）
        self.log_EI = X_val[:, 0]
        self.R = X_val[:, 1]
        
        # 誤差率 (%)
        # ゼロ除算防止
        self.rel_error = (self.y_pred - self.y_actual) / (self.y_actual + 1e-6) * 100.0

    def plot_dashboard(self):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("AI Surrogate Model Verification (Concentrated Layer Spec)", fontsize=16)

        # 1. Accuracy (実測 vs 予測)
        ax1 = fig.add_subplot(2, 3, 1)
        sc1 = ax1.scatter(self.y_actual, self.y_pred, c=np.abs(self.rel_error), cmap='coolwarm', s=10, vmin=0, vmax=2)
        ax1.plot([self.y_actual.min(), self.y_actual.max()], [self.y_actual.min(), self.y_actual.max()], 'k--', lw=1)
        ax1.set_xlabel('Actual Weight (kg/m)')
        ax1.set_ylabel('Predicted Weight (kg/m)')
        ax1.set_title('Accuracy Check')
        plt.colorbar(sc1, ax=ax1, label='Abs Error %')
        ax1.grid(True, alpha=0.3)

        # 2. Error Histogram
        ax2 = fig.add_subplot(2, 3, 2)
        mean_err = np.mean(self.rel_error)
        std_err = np.std(self.rel_error)
        ax2.hist(self.rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.axvline(mean_err, color='orange', linestyle='-', lw=2, label=f'Mean: {mean_err:.2f}%')
        ax2.set_title(f'Error Distribution\nMean: {mean_err:.2f}%, Std: {std_err:.2f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 3D Surface
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        # データが多いと重いので間引く
        idx = np.random.choice(len(self.y_actual), min(3000, len(self.y_actual)), replace=False)
        p = ax3.scatter(self.log_EI[idx], self.R[idx], self.y_actual[idx], 
                        c=self.rel_error[idx], cmap='bwr', s=5, vmin=-1, vmax=1)
        ax3.set_xlabel('Log10(EI)')
        ax3.set_ylabel('Diameter R')
        ax3.set_zlabel('Weight')
        ax3.set_title('Pareto Surface')
        fig.colorbar(p, ax=ax3, label='Error %', shrink=0.6)

        # 4. Bias vs Diameter
        ax4 = fig.add_subplot(2, 3, 4)
        sc4 = ax4.scatter(self.R, self.rel_error, c=self.log_EI, cmap='viridis', s=10, alpha=0.6)
        ax4.axhline(0, color='red', linestyle='--')
        ax4.set_xlabel('Diameter R (mm)')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title('Bias vs Diameter')
        plt.colorbar(sc4, ax=ax4, label='Log10(EI)')
        ax4.grid(True, alpha=0.3)

        # 5. Bias vs Stiffness
        ax5 = fig.add_subplot(2, 3, 5)
        sc5 = ax5.scatter(self.log_EI, self.rel_error, c=self.R, cmap='plasma', s=10, alpha=0.6)
        ax5.axhline(0, color='red', linestyle='--')
        ax5.set_xlabel('Log10(EI) [Stiffness]')
        ax5.set_ylabel('Relative Error (%)')
        ax5.set_title('Bias vs Stiffness')
        plt.colorbar(sc5, ax=ax5, label='Diameter R')
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ==========================================
# 3. メイン処理: AI学習
# ==========================================
def train_spar_surrogate():
    csv_path = "pareto_spar_dataset_user_model.csv"
    
    # 1. データ準備
    try:
        X, y, df = load_and_preprocess(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        print("Please run the previous data generation script first.")
        return

    # 2. 学習用とテスト用に分割 (8:2)
    # 本当に未知のデータに対し、汎化できているかを確認するため
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. モデル定義 (Random Forest)
    # パイプラインにしておくと後で便利
    model = Pipeline([
        ('rf', RandomForestRegressor(
            n_estimators=300,    # 木の数（多いほど安定）
            max_depth=None,      # 深さ制限なし（複雑なパレート境界に追従させる）
            min_samples_leaf=1,  # 細かい階段状の変化を許容
            random_state=42,
            n_jobs=-1            # 全力計算
        ))
    ])

    print("-" * 50)
    print("Training AI Model (Random Forest)...")
    model.fit(X_train, y_train)
    
    # 4. 評価
    print("Evaluating...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    print("-" * 50)
    print(f"Training R^2 : {train_score:.5f}")
    print(f"Test R^2     : {test_score:.5f} (Should be > 0.99)")
    print(f"Test MAE     : {mae:.5f} kg/m")
    print("-" * 50)

    # 5. モデル保存
    model_filename = "spar_weight_surrogate_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")

    # 6. 使い方の出力
    print("\n" + "="*50)
    print(" [Usage Guide] Copy this function to your Optimization Code:")
    print("="*50)
    print(f"""
import joblib
import numpy as np

# Load the AI model (Do this once)
spar_model = joblib.load('{model_filename}')

def predict_spar_weight(EI_target, Diameter_mm):
    \"\"\"
    Predicts the weight of an optimally designed spar (Concentrated Cap).
    Args:
        EI_target (float): Required Stiffness [Nmm^2]
        Diameter_mm (float): Spar Diameter [mm]
    Returns:
        float: Estimated Weight [kg/m]
    \"\"\"
    # Feature Engineering: Log10 transform for EI
    log_EI = np.log10(EI_target)
    
    # Predict (input must be 2D array)
    # Input format: [[LogEI, Diameter]]
    pred_w = spar_model.predict([[log_EI, Diameter_mm]])
    
    return pred_w[0]
    """)
    print("="*50)

    # 7. グラフによる最終確認 (テストデータを使用)
    evaluator = ModelEvaluator(X_test, y_test, model)
    evaluator.plot_dashboard()

if __name__ == "__main__":
    train_spar_surrogate()