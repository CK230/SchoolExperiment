import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 建立数据保存目录
os.makedirs('data/exp7_1', exist_ok=True)

# ==========================================
# 任务 A: 标准化数据环境搭建 [cite: 11]
# ==========================================
# 全局随机种子设定: 统一使用种子 42 [cite: 12]
np.random.seed(42)

def true_func(x):
    return np.sin(2 * np.pi * x) # 目标函数: y=sin(2πx) [cite: 13]

# 在[0,1]区间内取100个等间距点作为测试集 (不含噪声) [cite: 13]
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = true_func(x_test).flatten()

# 保存测试集 
pd.DataFrame({'x': x_test.flatten(), 'y': y_test}).to_csv('data/exp7_1/test_set.csv', index=False)

# ==========================================
# 任务 B: “平行宇宙”多采样拟合 [cite: 14]
# ==========================================
# 多数据集生成: 加入噪声 ε~N(0, 0.15^2)，生成50组训练集，每组20个样本 [cite: 15]
n_datasets = 50
n_samples = 20
noise_std = 0.15
lambdas = [1e-8, 1e-7, 1e-6, 1e-3, 0.01, 0.1, 1, 10] # 岭回归参数序列 [cite: 16]

all_predictions = {l: np.zeros((n_datasets, 100)) for l in lambdas}
all_train_data = []

for i in range(n_datasets):
    x_train = np.random.rand(n_samples, 1)
    y_train = true_func(x_train).flatten() + np.random.normal(0, noise_std, n_samples)
    
    # 岭回归批量拟合: 固定多项式阶数为10 [cite: 16, 17]
    for l in lambdas:
        model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=l))
        model.fit(x_train, y_train)
        all_predictions[l][i, :] = model.predict(x_test)
    
    # 记录数据集用于保留 
    df_temp = pd.DataFrame({'x': x_train.flatten(), 'y': y_train, 'dataset_id': i})
    all_train_data.append(df_temp)

pd.concat(all_train_data).to_csv('data/exp7_1/train_datasets.csv', index=False)

# ==========================================
# 任务 C: 定量计算与可视化 [cite: 18]
# ==========================================
bias_sq_list, var_list, total_error_list = [], [], []

for l in lambdas:
    preds = all_predictions[l]
    avg_pred = np.mean(preds, axis=0) # 平均预测曲线 [cite: 19]
    
    bias_sq = np.mean((avg_pred - y_test)**2) # 偏置平方 Bias^2 [cite: 19]
    variance = np.mean(np.var(preds, axis=0)) # 方差 Variance [cite: 19]
    total_error = np.mean((preds - y_test)**2) # 总误差 [cite: 19]
    
    bias_sq_list.append(bias_sq)
    var_list.append(variance)
    total_error_list.append(total_error)

# 核心曲线绘制: 以 log10(λ) 为横坐标 [cite: 21]
plt.figure(figsize=(8, 5))
plt.plot(np.log10(lambdas), bias_sq_list, 'r-o', label='$Bias^2$')
plt.plot(np.log10(lambdas), var_list, 'b-o', label='Variance')
plt.plot(np.log10(lambdas), total_error_list, 'g-o', label='Total Error')
plt.xlabel('$log_{10}(\lambda)$')
plt.ylabel('Error')
plt.legend()
plt.title('Bias-Variance Trade-off')
plt.show()

# 极端参数对比可视化 [cite: 22]
best_l = lambdas[np.argmin(total_error_list)]
for l_val in [1e-8, 10, best_l]:
    plt.figure(figsize=(6, 4))
    for i in range(n_datasets):
        plt.plot(x_test, all_predictions[l_val][i, :], color='gray', alpha=0.1)
    plt.plot(x_test, np.mean(all_predictions[l_val], axis=0), 'r', label='Mean Pred')
    plt.plot(x_test, y_test, 'g--', label='True Sine')
    plt.title(f'Lambda = {l_val}')
    plt.legend()
    plt.show()