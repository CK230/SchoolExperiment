# -*- coding: utf-8 -*-
"""
任务 A/B：构造严格共线性设计矩阵，比较正规方程（直接求逆）与岭回归的结果。
对应实验中的“正规方程的崩溃与岭回归”。
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

def build_collinear_data(n=100, seed=42):
    """
    构造严格共线性数据：
    x3 = x1 + x2
    真实系数 w = [1, 2, 3]
    y = Xw
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2  # 严格线性相关

    X = np.column_stack([x1, x2, x3])
    w_true = np.array([1.0, 2.0, 3.0])
    y = X @ w_true
    return X, y, w_true

def normal_equation_inverse(X, y):
    """
    正规方程：w = (X^T X)^(-1) X^T y
    当 X^T X 奇异时，这里会报错。
    """
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.inv(XtX) @ Xty
    return w

def ridge_regression_closed_form(X, y, lam=1e-3):
    """
    岭回归闭式解：
    w = (X^T X + λI)^(-1) X^T y
    """
    n_features = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX + lam * np.eye(n_features), Xty)
    return w

def mse(a, b):
    return np.mean((a - b) ** 2)

def main():
    # 1. 构造数据
    X, y, w_true = build_collinear_data()
    print("真实系数 w_true =", w_true)
    print("X 的形状 =", X.shape)
    print("rank(X) =", np.linalg.matrix_rank(X))
    print("rank(X^T X) =", np.linalg.matrix_rank(X.T @ X))
    print()

    # 2. 直接使用正规方程（尝试求逆）
    try:
        w_ne = normal_equation_inverse(X, y)
        print("正规方程求得的 w =", w_ne)
        print("与真实系数的 MSE =", mse(w_ne, w_true))
        print("正规方程误差范数 ||w - w_true||_2 =", np.linalg.norm(w_ne - w_true))
    except np.linalg.LinAlgError as e:
        print("正规方程失败：", e)
        print("原因：X^T X 为奇异矩阵，无法直接求逆。")

    print()

    # 3. 岭回归
    lam = 1e-3
    w_ridge = ridge_regression_closed_form(X, y, lam=lam)
    print(f"岭回归 (lambda={lam}) 求得的 w =", w_ridge)
    print("与真实系数的 MSE =", mse(w_ridge, w_true))
    print("岭回归误差范数 ||w - w_true||_2 =", np.linalg.norm(w_ridge - w_true))

    print()
    print("说明：由于第三列严格等于前两列之和，X 存在严格共线性，正规方程会失稳或直接失败；")
    print("岭回归通过加入 λI 提升了矩阵可逆性，因此通常能稳定求解。")

if __name__ == "__main__":
    main()
