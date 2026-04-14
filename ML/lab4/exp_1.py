import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import platform

# ===================== 修复点1：Matplotlib中文显示配置 =====================
sys_name = platform.system()
if sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows使用黑体
elif sys_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac使用Arial Unicode
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux fallback
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ===================== 实验1配置与初始化 =====================
np.random.seed(42)
data_save_dir = 'exp04_data'
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
print("="*70)
print("实验1：线性气温预测模型的解析解实现")
print("="*70)
print(f"数据集将保存至: {os.path.abspath(data_save_dir)}")

# ===================== 步骤A：构建合成数据集 =====================
N = 100
x1 = np.random.uniform(0, 2000, size=N)
x2 = np.random.uniform(0, 90, size=N)
X = np.c_[np.ones(N), x1, x2]

# ===================== 步骤B：生成观测标签y =====================
w_true = np.array([30.0, -0.006, -0.2]).reshape(-1, 1)
epsilon = np.random.normal(0, 1, size=(N, 1))
y = X @ w_true + epsilon

# ===================== 修复点2：数据集导出（指定UTF-8-BOM编码） =====================
exp1_data = np.column_stack((x1, x2, y.flatten()))
exp1_header = "海拔_x1,纬度_x2,气温_y"
exp1_data_path = os.path.join(data_save_dir, 'exp01_altitude_latitude_temp.csv')
# 新增 encoding='utf-8-sig'，Excel打开中文不乱码
np.savetxt(exp1_data_path, exp1_data, delimiter=',', header=exp1_header, comments='', fmt='%.6f', encoding='utf-8-sig')

exp1_weight_path = os.path.join(data_save_dir, 'exp01_true_weights.csv')
np.savetxt(exp1_weight_path, w_true, delimiter=',', header="真实权重_w", comments='', fmt='%.6f', encoding='utf-8-sig')

print(f"\n实验1数据集已保存：")
print(f"  - 特征与标签: {exp1_data_path}")
print(f"  - 真实权重: {exp1_weight_path}")

# ===================== 步骤C：满秩性验证 =====================
rank_X = np.linalg.matrix_rank(X)
XTX = X.T @ X
rank_XTX = np.linalg.matrix_rank(XTX)

print("\n" + "="*60)
print(f"设计矩阵X的形状: {X.shape}，秩: {rank_X}，列数: {X.shape[1]}")
print(f"X^T X的秩: {rank_XTX}")
if rank_X == X.shape[1]:
    print("验证结果：X列满秩，X^T X可逆，正规方程有唯一解")
else:
    print("验证结果：X非列满秩，X^T X不可逆，正规方程无唯一解")
print("="*60)

# ===================== 步骤D：正规方程求解与评估 =====================
w_hat = np.linalg.inv(XTX) @ X.T @ y
print(f"真实权重w_true:  {w_true.flatten().round(6)}")
print(f"估计权重w_hat:  {w_hat.flatten().round(6)}")
euclidean_dist = np.linalg.norm(w_hat - w_true)
print(f"w_hat与w_true的欧几里得距离: {euclidean_dist:.6f}")

# ===================== 3D可视化（中文现在可正常显示） =====================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='steelblue', s=40, label='原始采样数据', alpha=0.7)
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 2000, 100), np.linspace(0, 90, 100))
y_pred_grid = w_hat[0] + w_hat[1] * x1_grid + w_hat[2] * x2_grid
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, cmap='viridis', alpha=0.4, label='预测超平面')
ax.set_xlabel('海拔 x1', fontsize=10)
ax.set_ylabel('纬度 x2', fontsize=10)
ax.set_zlabel('气温 y', fontsize=10)
ax.set_title('海拔-纬度-气温 线性回归预测超平面', fontsize=12)
ax.legend()
plt.show()

# ===================== 步骤E：几何解释-正交投影验证 =====================
y_hat = X @ w_hat
r = y - y_hat
print("\n残差向量r与设计矩阵X各列的点积（正交性验证，结果应接近0）：")
for i in range(X.shape[1]):
    dot_product = np.dot(r.flatten(), X[:, i])
    col_name = "偏置项全1列" if i == 0 else f"特征x{i}列"
    print(f"  与{col_name}的点积: {dot_product:.10f}")
print("\n几何结论：残差与X的列空间正交，证明预测值y_hat是真值y在X列空间上的正交投影")