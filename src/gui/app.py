import customtkinter as ctk
import sys
import os

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- src/gui/app.py の冒頭部分 ---
import customtkinter as ctk
import sys
import os
import numpy as np # 必要に応じて

# プロジェクトルートへのパス通し
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 【重要】エラー回避のためのダミー定義
# エラーが PhysicsFeatureEngineer を探しているので、ここで定義します
def add_physics_features(X):
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    return np.column_stack((X, thickness_index))

# モデル保存時の形式に合わせてクラスとして定義
class PhysicsFeatureEngineer:
    def transform(self, X):
        return add_physics_features(X)

# その後でインポート
from src.optimization.snap_optimizer import SnapOptimizer

# 既存のロジックをインポート
from src.optimization.snap_optimizer import SnapOptimizer

class HemereGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # アプリの基本設定
        self.title("Hemere - HPA Spar Structural AI")
        self.geometry("600x500")
        ctk.set_appearance_mode("dark") # ダークモード設定
        ctk.set_default_color_theme("blue")

        # ロジックの初期化
        self.optimizer = SnapOptimizer()

        # --- レイアウト作成 ---
        self.header_label = ctk.CTkLabel(self, text="主翼桁 構造最適化エンジン", font=("Yu Gothic", 24, "bold"))
        self.header_label.pack(pady=20)

        # 入力エリア
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=10, padx=20, fill="x")

        self.label_ei = ctk.CTkLabel(self.input_frame, text="目標剛性 Target EI [Nmm^2]:")
        self.label_ei.grid(row=0, column=0, padx=10, pady=10)

        self.entry_ei = ctk.CTkEntry(self.input_frame, placeholder_text="例: 1.0e10")
        self.entry_ei.grid(row=0, column=1, padx=10, pady=10)

        # 計算実行ボタン
        self.calc_button = ctk.CTkButton(self, text="設計最適化を実行", command=self.run_optimization)
        self.calc_button.pack(pady=20)

        # 結果表示エリア
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.result_text = ctk.CTkTextbox(self.result_frame, font=("Consolas", 14))
        self.result_text.pack(pady=10, padx=10, fill="both", expand=True)

    def run_optimization(self):
        """SnapOptimizerを呼び出して結果を表示する"""
        try:
            target_ei = float(self.entry_ei.get())
            
            # 既存のsolveメソッドを実行
            res = self.optimizer.solve(target_ei)
            
            self.result_text.delete("1.0", "end")
            if res:
                output = (
                    f"【最適設計結果】\n"
                    f"----------------------------------\n"
                    f"推奨直径 (D)  : {res['Diameter']} mm\n"
                    f"推定重量 (W)  : {res['Actual_Weight']:.4f} kg/m\n"
                    f"実効剛性 (EI) : {res['Actual_EI']:.2e}\n"
                    f"剛性余力      : +{res['Margin_Pct']:.1f} %\n"
                    f"積層構成      : {res['Ply_Config']}\n"
                    f"----------------------------------\n"
                    f"※AIナビゲーションと物理スナップ完了"
                )
                self.result_text.insert("0.0", output)
            else:
                self.result_text.insert("0.0", "エラー: 探索範囲内に解が見つかりませんでした。")
        
        except ValueError:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("0.0", "エラー: 正しい数値を入力してください（例: 1.0e10）")

if __name__ == "__main__":
    app = HemereGUI()
    app.mainloop()