import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from tqdm import tqdm  # 進捗バー用のライブラリ

# --- 1. 物理パラメータの設定 ---
L = 10.0
N = 100
dx = L / N
rho_a = 0.5
m_node = rho_a * dx
c_val = 0.2

# --- 2. スパン方向の剛性 EI(x) の定義 ---
EI_root = 1.0e6
EI_tip = 5.0e3
EI_distribution = np.linspace(EI_root, EI_tip, N)

# --- 3. 剛性行列 K の作成 ---
num_active = N - 1
K_active = np.zeros((num_active, num_active))
for i in range(num_active):
    idx = i + 1 
    ei_local = EI_distribution[idx]
    if idx == 1:
        K_active[i, 0:3] = np.array([7, -4, 1]) * (ei_local / dx**4)
    elif idx == 2:
        K_active[i, 0:4] = np.array([-4, 6, -4, 1]) * (ei_local / dx**4)    
    elif idx == N-2:
        K_active[i, -4:] = np.array([1, -4, 5, -2]) * (ei_local / dx**4)
    elif idx == N-1:
        K_active[i, -3:] = np.array([1, -2, 1]) * (ei_local / dx**4)
    else:
        K_active[i, i-2:i+3] = np.array([1, -4, 6, -4, 1]) * (ei_local / dx**4)

# --- 4. 運動方程式の定義 ---
def get_total_load(t):
    rate = 800000.0
    max_load = 1000.0
    return min(rate * t, max_load)

# 進捗バーの初期化
pbar = tqdm(total=100, desc="Solving ODE", unit="%")
last_p = 0

def beam_dynamics(t, state):
    global last_p
    # 進捗率の更新 (t_stopに対する現在のtの割合)
    current_p = int(100 * t / 10.0)
    if current_p > last_p:
        pbar.update(current_p - last_p)
        last_p = current_p
        
    y = state[:num_active]
    v = state[num_active:]
    W = get_total_load(t)
    q_root = (4 * W) / (np.pi * L)
    
    f = []
    for idx in range(1, N):
        x = idx * dx
        q_x = q_root * np.sqrt(max(0, 1 - (x/L)**2))
        f.append(q_x * dx)
    f = np.array(f)
    
    accel = (f - c_val*v - (K_active @ y) * dx) / m_node
    return np.concatenate([v, accel])

# --- 5. シミュレーション実行 ---
t_stop = 5.0
fps = 60
t_eval = np.linspace(0, t_stop, int(t_stop * fps))

# solve_ivpを実行
# --- 5. シミュレーション実行 ---
# method='Radau' を指定することで、硬い方程式でも高速に解けるようになります
sol = solve_ivp(beam_dynamics, (0, t_stop), np.zeros(2*num_active), 
                t_eval=t_eval, method='Radau')
pbar.close() # 計算終了！

# --- 【重要】ここが NameError の原因箇所です：せん断力の履歴計算 ---
print("Calculating shear force history...")
shear_history = []
for i in range(len(sol.t)):
    t = sol.t[i]
    v_vec = sol.y[num_active:, i]
    # 加速度を再計算
    d_state = beam_dynamics(t, sol.y[:, i])
    accel = d_state[num_active:]
    
    W = get_total_load(t)
    # 反力の計算： V = 外力 - 慣性力 - 減衰力
    V_root = W - np.sum(m_node * accel) - np.sum(c_val * v_vec)
    shear_history.append(V_root)

# --- 6. 同時プロットのアニメーション設定 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)
# (上) 翼のしなりアニメーション
ax1.set_xlim(0, L)
ax1.set_ylim(-4.5, 0.5)
ax1.set_title("Dynamic Wing Deflection")
ax1.set_ylabel("Displacement [m]")
ax1.grid(True, alpha=0.3)
line_wing, = ax1.plot([], [], 'o-', color='royalblue', markersize=3, lw=2)

# (下) せん断力のリアルタイム推移
ax2.set_xlim(0, t_stop)
ax2.set_ylim(0, 1500)
ax2.set_title("Root Shear Force History")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Shear Force [N]")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Target 200N')
line_shear, = ax2.plot([], [], color='teal', lw=2)
point_shear, = ax2.plot([], [], 'o', color='teal') # 現在地を示す点
ax2.legend(loc='upper right')

def update(i):
    # 翼のデータ更新
    x_wing = np.linspace(0, L, N)
    y_wing = np.insert(-sol.y[:num_active, i], 0, 0) # 根元(0,0)を追加して下向きに
    line_wing.set_data(x_wing, y_wing)
    
    # せん断力のデータ更新
    line_shear.set_data(sol.t[:i], shear_history[:i])
    point_shear.set_data([sol.t[i]], [shear_history[i]])
    
    return line_wing, line_shear, point_shear

ani = FuncAnimation(fig, update, frames=len(sol.t), blit=True, interval=1000/fps)
plt.show()