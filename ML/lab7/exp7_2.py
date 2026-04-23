import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# 建立数据保存目录
os.makedirs('data/exp7_2', exist_ok=True)

# ==========================================
# 任务 A: 基础数据与离群点构造 [cite: 25]
# ==========================================
np.random.seed(42)
n_pts = 50
# 构建基础分布: 类别0(均值2), 类别1(均值3), 方差均 0.5^2 [cite: 26]
class0 = np.random.normal(2, 0.5, n_pts)
class1 = np.random.normal(3, 0.5, n_pts)

X_base = np.concatenate([class0, class1]).reshape(-1, 1)
y_base = np.concatenate([np.zeros(n_pts), np.ones(n_pts)])

# 引入极远距离离群点: 在 x=20 处添加标签为1的样本 [cite: 27]
X_outlier = np.vstack([X_base, [[20]]])
y_outlier = np.append(y_base, 1)

# 保存数据集 
pd.DataFrame({'x': X_outlier.flatten(), 'y': y_outlier}).to_csv('data/exp7_2/robustness_data.csv', index=False)

# ==========================================
# 任务 B: 模型训练与决策边界追踪 [cite: 28]
# ==========================================
# 回归模型拟合 [cite: 29]
model_base = LinearRegression().fit(X_base, y_base)
model_out = LinearRegression().fit(X_outlier, y_outlier)

# 决策边界定位: 求解 h(x)=0.5 对应的边界位置 x_db [cite: 30]
def get_db(model):
    return (0.5 - model.intercept_) / model.coef_[0]

db_base = get_db(model_base)
db_out = get_db(model_out)

# ==========================================
# 任务 C: 偏移评价与“误伤”检测 [cite: 31]
# ==========================================
# 边界偏移量计算 [cite: 32]
print(f"原始决策边界: {db_base:.4f}")
print(f"含离群点决策边界: {db_out:.4f}")
print(f"边界移动距离: {abs(db_out - db_base):.4f}")

# 样本误伤检查: 检查原本被正确分类的样本是否被划分为类别0 [cite: 33]
# 类别1的点如果小于新的边界 db_out，则被误判为0
misclassified = class1[class1 < db_out]
print(f"类别1被‘误伤’为类别0的样本数: {len(misclassified)}")

# 可视化结果
plt.figure(figsize=(10, 5))
plt.scatter(class0, np.zeros(n_pts), color='blue', label='Class 0')
plt.scatter(class1, np.ones(n_pts), color='orange', label='Class 1')
plt.scatter([20], [1], color='red', marker='x', s=100, label='Outlier (x=20)')

# 绘制决策边界线
plt.axvline(db_base, color='blue', linestyle='--', label=f'Original DB ({db_base:.2f})')
plt.axvline(db_out, color='red', linestyle='--', label=f'Shifted DB ({db_out:.2f})')
plt.title('Robustness of OLS: Boundary Shift Analysis')
plt.legend()
plt.show()