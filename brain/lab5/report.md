# 实验5 神经可塑性与编码应用实验报告



## 实验目的

1. **理解编码与可塑性的衔接关系：** 了解神经系统中的输入信息如何先被编码为脉冲表示，再在神经元与突触层面发生可塑性变化。
2. **掌握频率编码在图像任务中的基本应用：** 通过对 MNIST 图像进行均匀编码，观察输入灰度与脉冲计数之间的对应关系。
3. **理解赫布规则的基本思想：** 通过一个简化的两输入一输出模型观察“共同活动导致连接增强”的现象。
4. **理解不同编码方式在噪声下的差异：** 在统一脉冲域噪声链下，比较频率编码和首次脉冲发放时间编码的重构变化与误差曲线。
5. **继续掌握基础仿真与绘图方法：** 使用 numpy、matplotlib、torch 和 torchvision 编写简单实验程序，并通过图像与曲线分析实验结果。



## 实验结果与过程分析

### 一 MNIST 图像的频率编码入门



<img src="D:\Typoraimages\exp5_1.png" alt="exp5_1" style="zoom:50%;" />



修改 T = 1后可以观察到

<img src="D:\Typoraimages\exp5_1t=1.png" alt="exp5_1t=1" style="zoom:50%;" />

编码后的脉冲计数图是否仍然保留了原始数字的主要结构？当时间窗 T 变大或变小时，编码图会发生什么变化？

_答_:根据实验结果可以看到编码确实保留了原始数据结果。当T越小时信息损失比较严重，低灰度区域可能没有脉冲显示。



### 二 赫布规则突触增强

<img src="D:\Typoraimages\exp5_2.png" alt="exp5_2" style="zoom:50%;" />

改变eta

<img src="D:\Typoraimages\exp5_2etalow.png" alt="exp5_2etalow" style="zoom:50%;" />

<img src="D:\Typoraimages\exp5_2etahigh.png" alt="exp5_2etahigh" style="zoom:50%;" />

改变Threhold



<img src="D:\Typoraimages\exp5_2Threholdhigh.png" alt="exp5_2Threholdhigh" style="zoom:50%;" />

<img src="D:\Typoraimages\exp5_2Threholdlow.png" alt="exp5_2Threholdlow" style="zoom:50%;" />

改变ProA（调低）

<img src="D:\Typoraimages\exp5_2probAlow.png" alt="exp5_2probAlow" style="zoom:50%;" />



改变ProB(调高)

<img src="D:\Typoraimages\exp5_2probBhigh.png" alt="exp5_2probBhigh" style="zoom:50%;" />

调高或调低阈值、输入 A/B 的发放概率后，输出脉冲数和权重演化会发生什么变化？

根据实验结果图显示

| 参数变化 | 输出脉冲数 | 权重演化                   |
| :------- | :--------- | :------------------------- |
| 阈值 ↑   | 减少       | 增长变慢，两路差距缩小     |
| 阈值 ↓   | 增加       | 增长加快，优势输入更快饱和 |
| probA ↑  | 增加       | wA 更强，wB 停滞           |
| probA ↓  | 减少       | wA 弱化，wB 可能反超       |
| probB ↑  | 增加       | wB 加速，可能追平 wA       |
| probB ↓  | 减少       | wB 停滞，wA 主导           |

### 三 噪声条件下不同编码方式的抗噪对比



![exp5_3](D:\Typoraimages\exp5_3.png)

增加噪声强度

![](D:\Typoraimages\exp5_3noisyhigh.png)

**噪声变化时重构图变化**

**频率编码**：逐渐出现颗粒感，结构仍可辨认，误差平缓上升。

**TTFS 编码**：噪声稍大即出现大量噪点，背景伪脉冲，结构迅速崩溃，误差陡升。

所以频率编码更加稳定，首次脉冲时间编码对于噪声更加敏感。



实验脚本的代码流程图：

```
加载 MNIST 图像 (数字7)

定义全局参数: T_WINDOW=100, TTFS_SPAN=80

函数 add_salt_pepper_spike_noise(脉冲列表, 噪声强度):
    p = 噪声强度/100
    随机删除每个脉冲 (概率 p)
    随机注入一个新脉冲 (概率 p)
    返回新脉冲列表

函数 process_rate_coding(图像, 噪声强度):
    对每个像素:
        理想脉冲个数 n = round(灰度 * T_WINDOW)
        生成均匀分布的理想脉冲时刻
        调用 add_salt_pepper_spike_noise
        对每个保留脉冲加高斯抖动(标准差=噪声强度/10)
        重构值 = 剩余脉冲数 / T_WINDOW
    返回重构图

函数 process_ttfs_coding(图像, 噪声强度):
    对每个像素:
        若灰度==0: 理想脉冲列表为空
        否则: 理想首次时刻 = TTFS_SPAN * (1 - 灰度)
        调用 add_salt_pepper_spike_noise
        对保留脉冲加高斯抖动
        取最小时刻作为首次脉冲时间
        重构值 = 1 - 首次脉冲时间/TTFS_SPAN (若无脉冲则为0)
    返回重构图

主程序:
    计算无噪声基线重构图 (clean_rate, clean_ttfs)
    预计算噪声强度 0~30 的 MAD 曲线
    创建交互窗口:
        子图1: 原始图像
        子图2: 频率编码重构 (动态更新)
        子图3: TTFS 编码重构 (动态更新)
        子图4: 两条误差曲线 + 垂直标记线 + 圆点标记
    添加滑块 (0~30)
    滑块回调函数:
        获取当前噪声强度
        调用 process_rate_coding 和 process_ttfs_coding
        更新子图2、3的数据
        更新曲线上的标记点和垂直线位置
```

### **实验报告中回答以下问题**



赫布规则会优先增强与输出共同活动更频繁的连接，而稳态可塑性会对整体活动水平进行约束。请结合实验 2 和实验 3 讨论：这两类机制分别在“选择性学习”和“整体稳定性”中扮演什么角色？如果系统中只有其中一种机制，网络可能分别出现什么偏差或缺陷？



**答 ：**赫布规则负责选择性学习，强化与输出共同激活的突触连接；稳态可塑性负责整体稳定性，约束网络活动与权重范围。

只有赫布规则会导致权重无限增长、网络过度兴奋失稳；只有稳态可塑性会让网络无法学习有效关联，失去选择性。



频率编码和     TTFS 各自把信息主要存放在脉冲序列的哪一类特征中？这种信息载体的差异为什么会导致它们对“脉冲删除、伪脉冲注入、时间抖动”三类扰动的抗性不同？



**答：**频率编码将信息存于脉冲总数 / 发放频率，TTFS 将信息存于首次脉冲时间。

频率编码不依赖单脉冲时序，对三类扰动抗性强；TTFS 仅靠首个脉冲时间，脉冲删除、注入、抖动都会直接破坏信息，敏感度更高。