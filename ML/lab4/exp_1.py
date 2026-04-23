# -*- coding: utf-8 -*-
"""
实验4-任务1：线性回归解析解（正规方程）
功能：生成海拔、纬度、气温合成数据并导出，手动实现正规方程求解。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. 固定随机种子
# -----------------------------
np.random.seed(42)  #

# -----------------------------
# 2. 构造合成数据集
# -----------------------------
N = 100
# 特征1：海拔 [0, 2000]
x1 = np.random.uniform(0, 2000, N)   
# 特征2：纬度 [0, 90]
x2 = np.random.uniform(0, 90, N)     

# 设计矩阵 X: 第一列为偏置项(全1)
X = np.column_stack([np.ones(N), x1, x2])

# 真值参数 w_true = [30.0, -0.006, -0.2]
w_true = np.array([30.0, -0.006, -0.2])

# 加入高斯噪声 eps ~ N(0, 1)
eps = np.random.normal(0, 1, N)

# 观测标签 y
y = X @ w_true + eps

# --- 新增：导出数据至 CSV ---
df1 = pd.DataFrame({
    'Bias_Term': X[:, 0],
    'Altitude_x1': x1,
    'Latitude_x2': x2,
    'Temperature_y': y
})
df1.to_csv('task1_temperature_data.csv', index=False, encoding='utf-8')
print("任务1：合成数据已成功导出至 'task1_temperature_data.csv'")

# -----------------------------
# 3. 满秩性验证
# -----------------------------
rank_X = np.linalg.matrix_rank(X)  #
print("\nX 的形状:", X.shape)
print("X 的秩:", rank_X)

if rank_X == X.shape[1]:
    print("结论：X 满秩，X^T X 可逆。")
else:
    raise ValueError("设计矩阵不满秩，无法使用正规方程求解。")

# -----------------------------
# 4. 正规方程求解
# w_hat = (X^T X)^(-1) X^T y
# -----------------------------
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y

print("\n真实参数 w_true:", w_true)
print("估计参数 w_hat:", w_hat)

# 参数误差（欧几里得距离）
param_distance = np.linalg.norm(w_hat - w_true)
print(f"\n||w_hat - w_true||_2 = {param_distance:.6f}")

# -----------------------------
# 5. 几何解释：残差与正交性验证
# -----------------------------
y_hat = X @ w_hat
r = y - y_hat

print("\n正交性验证 (r 与 X 各列的点积应该趋于 0):")
for j in range(X.shape[1]):
    dp = float(r @ X[:, j])
    print(f"r 与 X 第 {j} 列的点积: {dp:.10f}")

# -----------------------------
# 6. 可视化
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, s=28, alpha=0.8, label='Observed data', color='blue')

# 绘制预测平面
x1_grid = np.linspace(x1.min(), x1.max(), 30)
x2_grid = np.linspace(x2.min(), x2.max(), 30)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)
Yg = w_hat[0] + w_hat[1] * X1g + w_hat[2] * X2g
ax.plot_surface(X1g, X2g, Yg, alpha=0.3, color='orange')

ax.set_xlabel("Altitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Temperature")
plt.title("Task 1: Linear Regression Normal Equation")
plt.legend()
plt.show()