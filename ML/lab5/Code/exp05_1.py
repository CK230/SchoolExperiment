import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 3.1 图像数据标准化 - 任务A 
def ld(p):
    x, y = [], []
    for d, l in [('cat', 0), ('dog', 1)]:
        dp = os.path.join(p, d)
        if not os.path.exists(dp): continue
        for f in os.listdir(dp):
            ip = os.path.join(dp, f)
            try:
                # 缩放为32x32灰度图并归一化 
                i = Image.open(ip).convert('L').resize((32, 32))
                a = np.array(i) / 255.0
                x.append(a.flatten()) # 展平为1024维向量 
                y.append(l)
            except: pass
    return np.array(x), np.array(y)

# 3.2 线性模型基准测试 - 任务A 
def slv(x, y, l):
    n = x.shape[1]
    i = np.eye(n)
    i[0, 0] = 0 # 偏置项不参与正则化 
    w = np.linalg.inv(x.T @ x + l * i) @ x.T @ y
    return w

# 3.2 线性模型基准测试 - 任务B 
def ac(x, y, w):
    p = x @ w
    c = (p >= 0.5).astype(int) # 阈值设为0.5 
    return np.mean(c == y)

# 3.3 多项式基函数扩展 - 任务A 
def pl(x):
    # 手动实现2次项(平方项)特征映射，扩展至2048维 
    return np.hstack((x, x**2))

# 数据路径设置
p1, p2 = '../data/train', '../data/val'
x1, y1 = ld(p1)
x2, y2 = ld(p2)

# 3.2 任务A: 构建包含偏置项的设计矩阵 X 
b1, b2 = np.ones((x1.shape[0], 1)), np.ones((x2.shape[0], 1))
x1_1, x2_1 = np.hstack((b1, x1)), np.hstack((b2, x2)) # 1次项
x1_2, x2_2 = np.hstack((b1, pl(x1))), np.hstack((b2, pl(x2))) # 2次项

# 3.4 正则化强度扫描 - 任务A 
# 生成15个点：1个 10^-6，其余14个处于 10^-3 到 15 之间
ls = np.concatenate(([1e-6], np.linspace(1e-3, 15, 14)))
r1, r2, ws = [], [], []

print("开始训练与评估...")
for l in ls:
    # 训练并评估1次项模型 
    w1 = slv(x1_1, y1, l)
    t1, v1 = ac(x1_1, y1, w1), ac(x2_1, y2, w1)
    r1.append(v1)
    ws.append(w1[1:]) # 记录权重用于后续可视化
    
    # 训练并评估2次项模型 
    w2 = slv(x1_2, y1, l)
    t2, v2 = ac(x1_2, y1, w2), ac(x2_2, y2, w2)
    r2.append(v2)
    
    # 3.2 & 3.3 任务B: 输出训练与验证集准确率 [cite: 1, 2]
    print(f"Lambda: {l:.2e}")
    print(f"  1次项 - 训练集准确率: {t1:.4f}, 验证集准确率: {v1:.4f}")
    print(f"  2次项 - 训练集准确率: {t2:.4f}, 验证集准确率: {v2:.4f}")

# 3.4 正则化强度扫描 - 任务B: 绘制准确率曲线 
plt.figure(figsize=(10, 6))
plt.plot(ls, r1, marker='o', label='Linear (1st order)')
plt.plot(ls, r2, marker='s', label='Polynomial (2nd order)')
plt.xscale('symlog', linthresh=1e-3) # 优化坐标轴以看清 1e-6
plt.xlabel('Lambda')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy vs Lambda')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

# 3.5 权重分布可视化 - 任务A & B 
# 选取起始、中间和末尾点进行对比
idx = [0, 7, 14]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, j in enumerate(idx):
    m = ws[j][:1024].reshape(32, 32) # 恢复为32x32图像矩阵 
    im = ax[i].imshow(m, cmap='coolwarm')
    ax[i].set_title(f'Lambda = {ls[j]:.2e}')
    plt.colorbar(im, ax=ax[i])
plt.tight_layout()
plt.show()