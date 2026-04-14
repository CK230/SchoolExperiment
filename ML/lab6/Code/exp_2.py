# -*- coding: utf-8 -*-
"""
任务 A/B：在二维平面中绘制单位圆，并按 SVD 分解后的变换顺序依次可视化：
x -> V^T x -> Σ(V^T x) -> UΣ(V^T x)
同时追踪起始点 (1, 0) 的坐标变化。
对应实验中的“SVD 的几何灵魂”。
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

def unit_circle_points(num=400):
    t = np.linspace(0, 2 * np.pi, num)
    x = np.cos(t)
    y = np.sin(t)
    return np.vstack([x, y])  # shape = (2, num)

def main():
    # 选取一个可视化效果较明显的二维矩阵
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])

    # SVD 分解：A = U @ S @ Vt
    U, s, Vt = np.linalg.svd(A)
    S = np.diag(s)

    print("矩阵 A =\n", A)
    print("\nU =\n", U)
    print("\n奇异值 s =", s)
    print("\nV^T =\n", Vt)

    # 单位圆与起始点
    circle = unit_circle_points()
    p0 = np.array([1.0, 0.0])  # 起始点

    # 依次变换
    circle_vt = Vt @ circle
    circle_s = S @ circle_vt
    circle_u = U @ circle_s

    p1 = Vt @ p0
    p2 = S @ p1
    p3 = U @ p2

    print("\n起始点 p0 =", p0)
    print("经过 V^T 后 p1 =", p1)
    print("经过 Σ 后 p2 =", p2)
    print("经过 U 后 p3 =", p3)
    print("直接计算 A p0 =", A @ p0)

    # 绘图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(circle[0], circle[1], label="Unit Circle")
    axes[0].scatter([p0[0]], [p0[1]], color="red", zorder=3, label="(1,0)")
    axes[0].set_title("Original")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(circle_vt[0], circle_vt[1], label=r"$V^T$")
    axes[1].scatter([p1[0]], [p1[1]], color="red", zorder=3)
    axes[1].set_title(r"After $V^T$")
    axes[1].axis("equal")
    axes[1].grid(True)

    axes[2].plot(circle_s[0], circle_s[1], label=r"$\Sigma V^T$")
    axes[2].scatter([p2[0]], [p2[1]], color="red", zorder=3)
    axes[2].set_title(r"After $\Sigma$")
    axes[2].axis("equal")
    axes[2].grid(True)

    axes[3].plot(circle_u[0], circle_u[1], label=r"$U\Sigma V^T$")
    axes[3].scatter([p3[0]], [p3[1]], color="red", zorder=3)
    axes[3].set_title(r"After $U$")
    axes[3].axis("equal")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
