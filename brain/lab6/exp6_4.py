import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "SimHei",          # Windows 常用
    "PingFang SC",      # macOS 常用
    "Microsoft YaHei",  # 微软雅黑
    "Arial Unicode MS", # 通用
    "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"
FEATURE_NAMES = ["OverallQual", "GrLivArea", "GarageCars", "YearBuilt", "FullBath"]
TARGET_NAME = "SalePrice"

def download_house_price_csv(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "kaggle_house_pred_train.csv")
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(DATA_URL, file_path)
    return file_path

def prepare_house_price_data(csv_path, train_ratio=0.8, seed=7):
    df = pd.read_csv(csv_path)
    work_df = df[FEATURE_NAMES + [TARGET_NAME]].copy()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(work_df))
    split = int(len(indices) * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]
    train_df = work_df.iloc[train_idx].copy()
    val_df = work_df.iloc[val_idx].copy()
    feature_means = train_df[FEATURE_NAMES].mean()
    feature_stds = train_df[FEATURE_NAMES].std().replace(0, 1.0)
    train_features = train_df[FEATURE_NAMES].fillna(feature_means)
    val_features = val_df[FEATURE_NAMES].fillna(feature_means)
    train_features = (train_features - feature_means) / feature_stds
    val_features = (val_features - feature_means) / feature_stds
    train_X = train_features.to_numpy(dtype=np.float64)
    val_X = val_features.to_numpy(dtype=np.float64)
    train_y = np.log1p(train_df[TARGET_NAME].to_numpy(dtype=np.float64)).reshape(-1, 1)
    val_y = np.log1p(val_df[TARGET_NAME].to_numpy(dtype=np.float64)).reshape(-1, 1)
    return train_X, train_y, val_X, val_y

def iterate_minibatches(num_samples, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield indices[start:end]

def init_regression_params(input_dim, hidden_size, seed=7):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.01, (input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.normal(0, 0.01, (hidden_size, 1))
    b2 = np.zeros((1, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_regression(X, y, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    H = np.maximum(0, Z1)
    y_pred = np.dot(H, params["W2"]) + params["b2"]
    loss = np.mean((y_pred - y) ** 2)
    cache = {"X": X, "y": y, "Z1": Z1, "H": H, "y_pred": y_pred}
    return cache, loss

def backward_regression_and_update(cache, params, lr):
    bs = cache["X"].shape[0]
    dy_pred = 2.0 * (cache["y_pred"] - cache["y"]) / bs
    dW2 = np.dot(cache["H"].T, dy_pred)
    db2 = np.sum(dy_pred, axis=0, keepdims=True)
    dH = np.dot(dy_pred, params["W2"].T)
    dZ1 = dH * (cache["Z1"] > 0)
    dW1 = np.dot(cache["X"].T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2
    return params

def evaluate_regression_dataset(X, y, params):
    cache, loss = forward_regression(X, y, params)
    return loss, cache["y_pred"]

def train_regression_model(train_X, train_y, val_X, val_y, hidden_size, lr, num_epochs, batch_size, seed=7):
    input_dim = train_X.shape[1]
    params = init_regression_params(input_dim, hidden_size, seed)
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        for b_idx in iterate_minibatches(len(train_X), batch_size, seed + epoch):
            b_X = train_X[b_idx]
            b_y = train_y[b_idx]
            cache, _ = forward_regression(b_X, b_y, params)
            params = backward_regression_and_update(cache, params, lr)
        t_loss, _ = evaluate_regression_dataset(train_X, train_y, params)
        v_loss, _ = evaluate_regression_dataset(val_X, val_y, params)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
    return params, history

def recover_price(log_price):
    return np.expm1(log_price)

def draw_loss_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=2.0, label="训练集损失 (MSE)")
    ax.plot(epochs, history["val_loss"], color="#c1121f", linewidth=2.0, label="验证集损失 (MSE)")
    ax.set_xlabel("训练轮次 (Epoch)")
    ax.set_ylabel("均方误差 (Loss)")
    ax.set_title("图 1：模型训练与验证损失曲线")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def draw_prediction_compare_panel(ax, y_true, y_pred):
    ax.clear()
    y_true_price = recover_price(y_true.reshape(-1))
    y_pred_price = recover_price(y_pred.reshape(-1))
    order = np.argsort(y_true_price)
    sorted_true = y_true_price[order]
    sorted_pred = y_pred_price[order]
    sample_rank = np.arange(len(sorted_true))
    ax.plot(sample_rank, sorted_true, color="#2563eb", linewidth=2.2, label="真实房价 (Ground Truth)")
    ax.plot(sample_rank, sorted_pred, color="#c1121f", linewidth=2.0, label="模型预测房价 (Prediction)", alpha=0.8)
    ax.set_xlabel("验证集样本 (按房价从低到高排序)")
    ax.set_ylabel("房价 (美元)")
    ax.set_title("图 2：验证集真实房价与预测房价对比")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    hidden_size = 32
    lr = 0.05
    num_epochs = 120
    batch_size = 64
    seed = 7
    csv_path = download_house_price_csv(data_dir="./data")
    train_X, train_y, val_X, val_y = prepare_house_price_data(csv_path=csv_path, seed=seed)
    params, history = train_regression_model(train_X, train_y, val_X, val_y, hidden_size, lr, num_epochs, batch_size, seed)
    train_loss, _ = evaluate_regression_dataset(train_X, train_y, params)
    val_loss, val_pred = evaluate_regression_dataset(val_X, val_y, params)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))  
    fig.patch.set_facecolor("#f8f9fa")
    draw_loss_panel(axes[0], history)
    draw_prediction_compare_panel(axes[1], val_y, val_pred)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()