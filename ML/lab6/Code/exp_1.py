# -*- coding: utf-8 -*-
import numpy as np
import os

np.set_printoptions(precision=6, suppress=True)

def build_collinear_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2

    X = np.column_stack([x1, x2, x3])
    w_true = np.array([1.0, 2.0, 3.0])
    y = X @ w_true

    # ===== 保存为 CSV =====
    os.makedirs("data", exist_ok=True)
    data = np.column_stack([X, y])
    header = "x1,x2,x3,y"
    np.savetxt("data/exp1_collinear_data.csv", data, delimiter=",", header=header, comments="")

    return X, y, w_true

def normal_equation_inverse(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def ridge_regression_closed_form(X, y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def main():
    X, y, w_true = build_collinear_data()

    try:
        w_ne = normal_equation_inverse(X, y)
        print("w_ne =", w_ne)
    except np.linalg.LinAlgError as e:
        print(e)

    w_ridge = ridge_regression_closed_form(X, y)
    print("w_ridge =", w_ridge)

if __name__ == "__main__":
    main()
