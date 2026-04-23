# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

np.set_printoptions(precision=6, suppress=True)

def unit_circle_points(num=400):
    t = np.linspace(0, 2 * np.pi, num)
    return np.vstack([np.cos(t), np.sin(t)])

def main():
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])

    U, s, Vt = np.linalg.svd(A)
    S = np.diag(s)

    circle = unit_circle_points()
    p0 = np.array([1.0, 0.0])

    # ===== 保存为 CSV =====
    os.makedirs("data", exist_ok=True)

    np.savetxt("data/exp2_matrix_A.csv", A, delimiter=",")
    np.savetxt("data/exp2_U.csv", U, delimiter=",")
    np.savetxt("data/exp2_S.csv", S, delimiter=",")
    np.savetxt("data/exp2_Vt.csv", Vt, delimiter=",")

    circle_T = circle.T  # 转置为 (n,2) 方便CSV
    np.savetxt("data/exp2_circle.csv", circle_T, delimiter=",", header="x,y", comments="")
    np.savetxt("data/exp2_p0.csv", p0.reshape(1, -1), delimiter=",", header="x,y", comments="")

    # 原流程
    circle_vt = Vt @ circle
    circle_s = S @ circle_vt
    circle_u = U @ circle_s

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(circle[0], circle[1])
    axes[1].plot(circle_vt[0], circle_vt[1])
    axes[2].plot(circle_s[0], circle_s[1])
    axes[3].plot(circle_u[0], circle_u[1])
    plt.show()

if __name__ == "__main__":
    main()
