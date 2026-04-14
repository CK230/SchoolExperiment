import numpy as np
import matplotlib.pyplot as plt
import os
import platform

# ===================== 修复点1：Matplotlib中文显示配置 =====================
sys_name = platform.system()
if sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif sys_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 实验2配置与初始化 =====================
np.random.seed(42)
data_save_dir = 'exp04_data'
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
print("="*70)
print("实验2：多项式基底与模型复杂度分析")
print("="*70)
print(f"数据集将保存至: {os.path.abspath(data_save_dir)}")

# ===================== 步骤A：合成正弦数据集生成 =====================
N = 15
x = np.random.uniform(0, 1, size=N)
y_true = np.sin(2 * np.pi * x)
epsilon = np.random.normal(0, 0.1, size=N)
y = y_true + epsilon
x_plot = np.linspace(0, 1, 1000)
y_plot_true = np.sin(2 * np.pi * x_plot)

# ===================== 修复点2：数据集导出（指定UTF-8-BOM编码） =====================
exp2_data = np.column_stack((x, y, y_true))
exp2_header = "输入_x,带噪声输出_y,真实函数值_y_true"
exp2_data_path = os.path.join(data_save_dir, 'exp02_sine_wave_data.csv')
np.savetxt(exp2_data_path, exp2_data, delimiter=',', header=exp2_header, comments='', fmt='%.6f', encoding='utf-8-sig')

print(f"\n实验2数据集已保存：")
print(f"  - 正弦曲线数据: {exp2_data_path}")

# ===================== 步骤B：多项式设计矩阵构建函数 =====================
def build_poly_design_matrix(x, M):
    N = len(x)
    Φ = np.ones((N, M+1))
    for m in range(1, M+1):
        Φ[:, m] = x ** m
    return Φ

# 设计矩阵满秩性验证
Φ_test = build_poly_design_matrix(x, M=12)
rank_Φ = np.linalg.matrix_rank(Φ_test)
rank_ΦTΦ = np.linalg.matrix_rank(Φ_test.T @ Φ_test)
print("\n" + "="*60)
print(f"多项式设计矩阵Φ(M=12)的秩: {rank_Φ}，列数: {Φ_test.shape[1]}")
print(f"Φ^T Φ的秩: {rank_ΦTΦ}")
if rank_Φ == Φ_test.shape[1]:
    print("满秩验证通过：Φ列满秩，Φ^T Φ可逆，有唯一解析解")
else:
    print("满秩验证失败：Φ非列满秩，Φ^T Φ不可逆")
print("="*60)

# ===================== 步骤C：多阶数模型求解与可视化 =====================
M_list = [1, 3, 12]
colors = ['#ff4444', '#00C851', '#aa66cc']

plt.figure(figsize=(12, 8))
plt.scatter(x, y, c='steelblue', s=60, label='带噪声的训练数据', zorder=5)
plt.plot(x_plot, y_plot_true, 'k--', linewidth=2, label='真实函数: sin(2πx)')

for M, color in zip(M_list, colors):
    Φ = build_poly_design_matrix(x, M)
    ΦTΦ = Φ.T @ Φ
    w_hat = np.linalg.inv(ΦTΦ) @ Φ.T @ y
    Φ_plot = build_poly_design_matrix(x_plot, M)
    y_plot_pred = Φ_plot @ w_hat
    plt.plot(x_plot, y_plot_pred, color=color, linewidth=2, label=f'拟合曲线 M={M}')
    print(f"\nM={M} 拟合结果：")
    print(f"  权重w_hat: {w_hat.round(4)}")
    print(f"  权重L2范数: {np.linalg.norm(w_hat):.4f}")

plt.xlabel('输入 x', fontsize=11)
plt.ylabel('输出 y', fontsize=11)
plt.ylim(-1.5, 1.5)
plt.title('不同多项式阶数M的线性回归拟合效果', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*60)
print("过拟合分析结论：")
print("1. M=1：欠拟合，模型复杂度不足，无法拟合正弦非线性规律")
print("2. M=3：拟合效果最优，贴合真实函数，泛化能力好")
print("3. M=12：过拟合，权重范数爆炸，曲线剧烈震荡，完美拟合训练点但完全偏离真实函数")