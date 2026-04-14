# 机器学习上机实验06：线性回归-3

## 实验目标 

1. 数值稳定性直观感知：通过构造共线性数据，观察正规方程在处理奇异矩阵时的失效现象。 
2. SVD 几何意义理解：通过坐标变换可视化，掌握奇异值分解中旋转与拉伸的物理含义。 
3. 矩阵病态性分析：学习条件数概念，理解观测噪声如何在高共线性场景下被放大，并掌握岭回归的 缓解机制。

## 实验环节

###  (Singularity & Ridge)

**任务 A：构造共线性设计矩阵：**  利用 ```numpy``` 构造$100 \times 3$的矩阵 $X$ , 确保第三列 $x_3$ 严格等于前两列之和。设定真实系数 $w = [1， 2， 3]$ 生成标签 $y$ 。

```python
def build_collinear_data(n=100, seed=42):
	rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2  # 严格线性相关

    X = np.column_stack([x1, x2, x3])
    w_true = np.array([1.0, 2.0, 3.0])
    y = X @ w_true
    return X, y, w_true
```



**任务B：算法对比**

```python
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
```

算法求解分别为正规方程和岭回归。

实验结果：

<img src="D:\Typoraimages\image-20260414163557795.png" alt="image-20260414163557795" style="zoom:50%;" />

**现象:**

在构造严格共线性数据（$x_3 = x_1 + x_2$）后，设计矩阵秩下降；

正规方程：得到明显偏离真实值 $[1,2,3]$ 的异常解
岭回归仍能得到稳定解，且与真实系数接近

**原因:**

共线性导致：
$$
X^T X \text{ 不可逆（奇异矩阵）}
$$

正规方程依赖矩阵求逆，因此数值不稳定；岭回归通过引入正则项：

$$
X^T X + \lambda I
$$
使矩阵变为正定，从而恢复可逆性。

**结论:**

共线性会导致线性回归解的不唯一性与数值不稳定，正规方程在该情况下不可靠，岭回归通过正则化有效提高求解稳定性，是处理共线性问题的常用方法。



### Geometric Interpretation of SVD

**任务A**：

```python
    # SVD 分解：A = U @ S @ Vt
    U, s, Vt = np.linalg.svd(A)
    S = np.diag(s)
```

```python
# 依次变换
    circle_vt = Vt @ circle
    circle_s = S @ circle_vt
    circle_u = U @ circle_s
```

**实验结果**如下：

![image-20260414163939660](D:\Typoraimages\image-20260414163939660.png)

![image-20260414164254925](D:\Typoraimages\image-20260414164254925.png)

**红点轨迹分析**：

**现象**
单位圆在变换过程中经历：

保持形状但发生旋转

被拉伸为椭圆

再次旋转形成最终形状

起始点 $(1,0)$ 的轨迹表现为方向变化与长度变化的叠加

**原因:**
SVD 分解：
$$
A = U \Sigma V^T
$$
对应三个几何操作：
$V^T$：对输入空间进行旋转
$\Sigma$：沿主轴方向进行缩放（由奇异值决定）
$U$：对结果再进行旋转
**结论**
任意线性变换均可分解为“旋转 + 缩放 + 旋转”,奇异值刻画了不同方向上的伸缩程度



### 病态问题与条件数分析 (Condition Number & Stability)

**任务A：**

构造病态矩阵

```python
def build_ill_conditioned_data(n=100, seed=7, eta=1e-8):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = x1 + eta * rng.normal(size=n)
    x3 = x1 + x2 + eta * rng.normal(size=n)

    X = np.column_stack([x1, x2, x3])
    w_true = np.array([1.0, 2.0, 3.0])
    y = X @ w_true
    return X, y, w_true

```
条件数分析
```python
    s_X = np.linalg.svd(X, compute_uv=False)
    s_XtX = np.linalg.svd(XtX, compute_uv=False)

    cond_X = s_X[0] / s_X[-1]
    cond_XtX = s_XtX[0] / s_XtX[-1]
```
**任务B**

无噪声基准

```	python
    try:
        w_clean = normal_equation_solution(X, y)
        print("无噪声标签下，正规方程解 w_clean =", w_clean)
        print("||w_clean - w_true||_2 =", np.linalg.norm(w_clean - w_true))
    except np.linalg.LinAlgError as e:
        print("无噪声情况下正规方程失败：", e)
```

有噪声

```python
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
```

**任务C**

岭回归稳定性的验证：

```python
    lam = 1e-3
    w_ridge_clean = ridge_solution(X, y, lam=lam)
    w_ridge_noisy = ridge_solution(X, y_noisy, lam=lam)
```



**实验结果**：

<img src="D:\Typoraimages\image-20260414165042930.png" alt="image-20260414165042930" style="zoom:50%;" />

**结论**：当$X$ 接近共线时，$X^T X$ 的条件数会显著增大，这会放大标签噪声对参数估计的影响；岭回归通过正则化通常能显著缓解这一问题。





