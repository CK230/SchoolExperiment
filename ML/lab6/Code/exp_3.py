# -*- coding: utf-8 -*-
"""
任务 A/B/C：
1) 构造近似病态矩阵（加入极小扰动）
2) 计算 X^T X 的条件数，并验证其与奇异值的关系
3) 给标签加入微小噪声，观察正规方程解的敏感性
4) 使用岭回归重新求解并比较稳定性

对应实验中的“病态问题与条件数分析”。
"""

import numpy as np

np.set_printoptions(precision=8, suppress=True)

def build_ill_conditioned_data(n=100, seed=7, eta=1e-8):
    """
    构造近似共线的数据矩阵：
    x2 = x1 + 小扰动
    x3 = x1 + x2 + 小扰动
    这样 X 会接近病态，但通常不至于完全奇异。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = x1 + eta * rng.normal(size=n)
    x3 = x1 + x2 + eta * rng.normal(size=n)

    X = np.column_stack([x1, x2, x3])
    w_true = np.array([1.0, 2.0, 3.0])
    y = X @ w_true
    return X, y, w_true

def normal_equation_solution(X, y):
    """
    直接使用正规方程求解。
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

def ridge_solution(X, y, lam=1e-3):
    """
    岭回归闭式解。
    """
    d = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX + lam * np.eye(d), Xty)

def main():
    # 1. 构造病态矩阵
    X, y, w_true = build_ill_conditioned_data(eta=1e-8)
    XtX = X.T @ X

    print("真实系数 w_true =", w_true)
    print("rank(X) =", np.linalg.matrix_rank(X))
    print()

    # 2. 条件数分析
    s_X = np.linalg.svd(X, compute_uv=False)
    s_XtX = np.linalg.svd(XtX, compute_uv=False)

    cond_X = s_X[0] / s_X[-1]
    cond_XtX = s_XtX[0] / s_XtX[-1]

    print("X 的奇异值 =", s_X)
    print("X 的条件数 cond(X) =", cond_X)
    print("X^T X 的奇异值 =", s_XtX)
    print("X^T X 的条件数 cond(X^T X) =", cond_XtX)
    print("理论上，cond(X^T X) 约等于 cond(X)^2")
    print("cond(X)^2 =", cond_X ** 2)
    print()

    # 3. 无噪声标签下的正规方程解
    try:
        w_clean = normal_equation_solution(X, y)
        print("无噪声标签下，正规方程解 w_clean =", w_clean)
        print("||w_clean - w_true||_2 =", np.linalg.norm(w_clean - w_true))
    except np.linalg.LinAlgError as e:
        print("无噪声情况下正规方程失败：", e)

    print()

    # 4. 给标签加入微小噪声，测试敏感性
    rng = np.random.default_rng(123)
    noise_level = 1e-4
    y_noisy = y + noise_level * rng.normal(size=y.shape)

    try:
        w_noisy = normal_equation_solution(X, y_noisy)
        print("加入噪声后，正规方程解 w_noisy =", w_noisy)
        print("||w_noisy - w_true||_2 =", np.linalg.norm(w_noisy - w_true))
        print("||w_noisy - w_clean||_2 =", np.linalg.norm(w_noisy - w_clean))
    except np.linalg.LinAlgError as e:
        print("加入噪声后正规方程失败：", e)

    print()

    # 5. 岭回归稳定性验证
    lam = 1e-3
    w_ridge_clean = ridge_solution(X, y, lam=lam)
    w_ridge_noisy = ridge_solution(X, y_noisy, lam=lam)

    print(f"岭回归 (lambda={lam}) 无噪声解 =", w_ridge_clean)
    print("||w_ridge_clean - w_true||_2 =", np.linalg.norm(w_ridge_clean - w_true))
    print()
    print(f"岭回归 (lambda={lam}) 噪声解 =", w_ridge_noisy)
    print("||w_ridge_noisy - w_true||_2 =", np.linalg.norm(w_ridge_noisy - w_true))
    print("||w_ridge_noisy - w_ridge_clean||_2 =", np.linalg.norm(w_ridge_noisy - w_ridge_clean))
    print()

    print("结论：当 X 接近共线时，X^T X 的条件数会显著增大，")
    print("这会放大标签噪声对参数估计的影响；岭回归通过正则化通常能显著缓解这一问题。")

if __name__ == "__main__":
    main()
