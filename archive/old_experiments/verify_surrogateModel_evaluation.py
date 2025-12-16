import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================
# 設定
# ==========================================
MODEL_PATH = "spar_weight_surrogate_model.pkl"

# 検証したい範囲（HPAのスパの一般的な範囲）
EI_MIN, EI_MAX = 1e9, 1e12  # Nmm^2
R_MIN, R_MAX = 40, 130      # mm (Diameter)

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return None
    print(f"Loading model from {MODEL_PATH}...")
    return joblib.load(MODEL_PATH)

def predict_weight(model, EI, R):
    """
    モデルを使って重量予測を行うラッパー関数
    学習時と同じ前処理(Log10変換)をここで行う
    """
    # 入力を配列化（単一値でも配列でも対応）
    EI = np.atleast_1d(EI)
    R = np.atleast_1d(R)
    
    # 特徴量作成: [Log10(EI), R]
    # 注意: 学習時と同じ順序・変換であること
    X = np.column_stack((np.log10(EI), R))
    
    return model.predict(X)

# def visualize_landscape(model):
#     """
#     設計空間全体のマップ（等高線図）を描画
#     これで「モデルの滑らかさ」を確認する
#     """
#     print("Generating Landscape map...")
    
#     # メッシュグリッド作成
#     ei_range = np.logspace(np.log10(EI_MIN), np.log10(EI_MAX), 100)
#     r_range = np.linspace(R_MIN, R_MAX, 100)
#     EI_grid, R_grid = np.meshgrid(ei_range, r_range)
    
#     # 予測実行（フラットにして投げる）
#     W_pred = predict_weight(model, EI_grid.ravel(), R_grid.ravel())
#     W_grid = W_pred.reshape(EI_grid.shape)
    
#     # --- Plotting ---
#     fig = plt.figure(figsize=(14, 6))
    
#     # 1. 2D Contour Plot (Heatmap)
#     ax1 = fig.add_subplot(1, 2, 1)
#     cp = ax1.contourf(EI_grid, R_grid, W_grid, levels=50, cmap='viridis')
#     fig.colorbar(cp, ax=ax1, label='Predicted Weight [kg/m]')
    
#     ax1.set_xscale('log') # EIはログスケールで表示
#     ax1.set_xlabel('Stiffness EI [Nmm^2]')
#     ax1.set_ylabel('Diameter R [mm]')
#     ax1.set_title('AI Prediction Landscape (Weight Map)')
#     ax1.grid(True, which="both", ls="-", alpha=0.3)
    
#     # 2. Cross Section (感度解析)
#     # 代表的なEIにおける「直径 vs 重量」のグラフ
#     ax2 = fig.add_subplot(1, 2, 2)
#     sample_EIs = [1e9, 5e9, 8e9, 1e10, 1e11]
    
#     for ei_val in sample_EIs:
#         r_line = np.linspace(R_MIN, R_MAX, 50)
#         w_line = predict_weight(model, np.full_like(r_line, ei_val), r_line)
#         ax2.plot(r_line, w_line, label=f'EI = {ei_val:.0e}')
        
#     ax2.set_xlabel('Diameter R [mm]')
#     ax2.set_ylabel('Predicted Weight [kg/m]')
#     ax2.set_title('Sensitivity Check: Diameter vs Weight')
#     ax2.legend()
#     ax2.grid(True, alpha=0.5)
    
#     plt.tight_layout()
#     plt.show()

# 追加で必要
from matplotlib.colors import LogNorm

def visualize_landscape_with_data(model, csv_path):
    """
    AIの予測曲面と、実際の学習データ(CSV)を重ねて表示する
    """
    print("Generating Landscape map with Raw Data...")
    
    # 1. 元データの読み込み
    df = pd.read_csv(csv_path)
    # 範囲外のデータがあってもプロットできるよう全データを取得
    actual_EI = df['EI'].values
    actual_R = df['R'].values
    actual_W = df['Weight'].values

    # 2. メッシュグリッド作成 (範囲はデータのMin-Maxに合わせるか、指定範囲にする)
    # データの範囲を確認
    print(f"Data EI Range: {actual_EI.min():.2e} ~ {actual_EI.max():.2e}")
    print(f"Data R  Range: {actual_R.min()} ~ {actual_R.max()}")

    # 指定範囲 (検証用)
    ei_range = np.logspace(np.log10(EI_MIN), np.log10(EI_MAX), 100)
    r_range = np.linspace(R_MIN, R_MAX, 100)
    EI_grid, R_grid = np.meshgrid(ei_range, r_range)
    
    # 3. 予測実行
    W_pred = predict_weight(model, EI_grid.ravel(), R_grid.ravel())
    W_grid = W_pred.reshape(EI_grid.shape)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 7))
    
    # 1. 2D Contour Plot (Heatmap) + Scatter
    ax1 = fig.add_subplot(1, 2, 1)
    
    # AIの予測（背景）
    cp = ax1.contourf(EI_grid, R_grid, W_grid, levels=50, cmap='viridis', alpha=0.6)
    fig.colorbar(cp, ax=ax1, label='AI Predicted Weight [kg/m]')
    
    # 元データ（点）を重ねる
    # 色を重量にして、AIの背景色と同じ傾向か確認
    sc = ax1.scatter(actual_EI, actual_R, c=actual_W, cmap='viridis', edgecolors='k', s=20, label='Training Data')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Stiffness EI [Nmm^2]')
    ax1.set_ylabel('Diameter R [mm]')
    ax1.set_title('AI Prediction vs Training Data Distribution')
    ax1.legend()
    
    # 2. Sensitivity Check (Slice)
    # データの存在するEI領域でスライスを切ってみる
    ax2 = fig.add_subplot(1, 2, 2)
    
    # データの中央値に近いEIを選ぶ
    target_EI = np.median(actual_EI)
    print(f"Slicing at EI approx {target_EI:.2e}")

    # AIの予測線
    r_line = np.linspace(R_MIN, R_MAX, 100)
    w_line = predict_weight(model, np.full_like(r_line, target_EI), r_line)
    ax2.plot(r_line, w_line, 'r-', linewidth=3, label=f'AI Prediction (EI={target_EI:.1e})')
    
    # そのEIに近い生データをプロットしてみる (±10%の範囲)
    mask = (actual_EI > target_EI * 0.9) & (actual_EI < target_EI * 1.1)
    if np.any(mask):
        ax2.scatter(actual_R[mask], actual_W[mask], alpha=0.6, s=30, c='blue', label='Actual Data (Nearby EI)')
    else:
        print("No data points found near target EI for slicing plot.")

    ax2.set_xlabel('Diameter R [mm]')
    ax2.set_ylabel('Weight [kg/m]')
    ax2.set_title(f'Slice Check at EI ~ {target_EI:.1e}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 実行部 (if __name__ == "__main__": の中) で呼び出す
# visualize_landscape_with_data(model, "pareto_spar_dataset_user_model.csv")

def interactive_check(model):
    """
    ユーザーが手動で値を入力して確認するモード
    """
    print("\n" + "="*40)
    print(" Interactive Mode (Ctrl+C to exit)")
    print("="*40)
    
    while True:
        try:
            ei_str = input(f"\nEnter EI [{EI_MIN:.0e} - {EI_MAX:.0e}]: ")
            if not ei_str: break
            ei_val = float(ei_str)
            
            r_str = input(f"Enter R  [{R_MIN} - {R_MAX}]: ")
            if not r_str: break
            r_val = float(r_str)
            
            weight = predict_weight(model, ei_val, r_val)[0]
            
            print(f" -> Predicted Weight: {weight:.4f} kg/m")
            
            # 簡易的な物理チェック
            if weight < 0:
                print(" [WARNING] Negative weight! Model is extrapolating badly.")
            
        except ValueError:
            print("Invalid input. Please enter numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    model = load_model()
    csvpath = "pareto_spar_dataset_user_model.csv"
    
    if model:
        # 1. 全体像の可視化
        visualize_landscape_with_data(model,csvpath)
        
        # 2. 手動チェックモード
        interactive_check(model)