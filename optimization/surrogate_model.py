import numpy as np

def estimate_spar_weight(EI, diameter_mm, safety_factor=1.03):
    """
    HPA用円筒桁の重量推定サロゲートモデル (Hemere Project)
    
    Args:
        EI (float): 曲げ剛性 [N*mm^2] ※単位注意！ ASBの出力がN*m^2なら 1e6 倍すること
        diameter_mm (float): 桁直径 [mm]
        safety_factor (float): 安全率 (デフォルト1.03 = +3%)
                               接着剤重量や製作誤差、プライ数の量子化誤差を吸収するため。

    Returns:
        float: 桁の線密度 [kg/m]
    """
    # 100万回のモンテカルロシミュレーション(Top 1.0% Elite)から導出された係数
    A = 1.890215e-09
    B = 0.826560
    C = -1.296560
    D = 0.000059

    # サロゲートモデル式: Weight = A * EI^B * R^C + D
    # ※EIは N*mm^2, diameter_mm は mm
    raw_weight = A * (EI ** B) * (diameter_mm ** C) + D

    # 安全率を考慮して返す
    return raw_weight * safety_factor

# --- テスト実行用 ---
if __name__ == "__main__":
    # テストケース: 直径80mm, EI=2.0e10 (典型的な値)
    test_EI = 2.0e10
    test_D = 80.0
    w = estimate_spar_weight(test_EI, test_D)
    print(f"Test Case:")
    print(f"  EI: {test_EI:.2e} N*mm^2")
    print(f"  D : {test_D} mm")
    print(f"  Estimated Weight: {w:.5f} kg/m")