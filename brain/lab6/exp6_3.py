import gzip
import os
import struct
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

CLASS_INFO = {
    1: "trouser",
    8: "bag",
    9: "ankle boot",
}
FASHION_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
FASHION_BACKUP_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

def ensure_fashion_backup_files(root):
    raw_dir = os.path.join(root, "FashionMNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for filename in FASHION_FILES:
        file_path = os.path.join(raw_dir, filename)
        if not os.path.exists(file_path):
            url = f"{FASHION_BACKUP_BASE}/{filename}"
            urllib.request.urlretrieve(url, file_path)
    return raw_dir

def read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, num_rows, num_cols).astype(np.float64) / 255.0
    return images

def read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(num_items)

def load_fashion_arrays_from_backup(root, train=True):
    raw_dir = ensure_fashion_backup_files(root)
    if train:
        image_file = "train-images-idx3-ubyte.gz"
        label_file = "train-labels-idx1-ubyte.gz"
    else:
        image_file = "t10k-images-idx3-ubyte.gz"
        label_file = "t10k-labels-idx1-ubyte.gz"
    images = read_idx_images_gz(os.path.join(raw_dir, image_file))
    labels = read_idx_labels_gz(os.path.join(raw_dir, label_file))
    return images, labels

def collect_balanced_subset_from_arrays(images, targets, selected_labels, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    label_map = {old_label: new_label for new_label, old_label in enumerate(selected_labels)}
    all_indices = []
    for old_label in selected_labels:
        class_indices = np.where(targets == old_label)[0]
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        all_indices.extend(chosen.tolist())
    rng.shuffle(all_indices)  
    all_indices = np.array(all_indices, dtype=int)
    old_targets = targets[all_indices]
    subset_images = images[all_indices].astype(np.float64)
    labels = np.array([label_map[int(old_label)] for old_label in old_targets], dtype=int)
    features = subset_images.reshape(len(subset_images), -1)
    return features, labels, subset_images

def load_fashion_subset(
    root="./data",
    train_samples_per_class=180,
    val_samples_per_class=60,
    selected_labels=(1, 8, 9),
    seed=7,
):
    transform = transforms.ToTensor()
    try:
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        train_images_all = train_dataset.data.numpy().astype(np.float64) / 255.0
        train_targets_all = train_dataset.targets.numpy()
        val_images_all = test_dataset.data.numpy().astype(np.float64) / 255.0
        val_targets_all = test_dataset.targets.numpy()
    except RuntimeError:
        train_images_all, train_targets_all = load_fashion_arrays_from_backup(root=root, train=True)
        val_images_all, val_targets_all = load_fashion_arrays_from_backup(root=root, train=False)
    train_X, train_y, train_images = collect_balanced_subset_from_arrays(
        train_images_all,
        train_targets_all, 
        selected_labels=selected_labels,
        samples_per_class=train_samples_per_class,
        seed=seed,
    )
    val_X, val_y, val_images = collect_balanced_subset_from_arrays(
        val_images_all,
        val_targets_all,
        selected_labels=selected_labels,
        samples_per_class=val_samples_per_class,
        seed=seed + 1,
    )
    label_names = [CLASS_INFO[label] for label in selected_labels]
    return train_X, train_y, train_images, val_X, val_y, val_images, label_names

def iterate_minibatches(num_samples, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield indices[start:end]

def init_params(input_dim, hidden_size, output_dim, seed=7):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.01, (input_dim, hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(0, 0.01, (hidden_size, output_dim))
    b2 = np.zeros(output_dim)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_pass(X, y, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    H = np.maximum(0, Z1)
    O = np.dot(H, params["W2"]) + params["b2"]
    
    O_max = np.max(O, axis=1, keepdims=True)
    exp_O = np.exp(O - O_max)
    Y_prob = exp_O / np.sum(exp_O, axis=1, keepdims=True)
    
    bs = X.shape[0]
    loss = -np.sum(np.log(Y_prob[np.arange(bs), y] + 1e-8)) / bs
    
    cache = {"X": X, "y": y, "Z1": Z1, "H": H, "Y_prob": Y_prob}
    return cache, loss

def backward_and_update(cache, params, lr):
    bs = cache["X"].shape[0]
    dO = cache["Y_prob"].copy()
    dO[np.arange(bs), cache["y"]] -= 1
    dO /= bs
    
    dW2 = np.dot(cache["H"].T, dO)
    db2 = np.sum(dO, axis=0)
    
    dH = np.dot(dO, params["W2"].T)
    dZ1 = dH * (cache["Z1"] > 0)
    
    dW1 = np.dot(cache["X"].T, dZ1)
    db1 = np.sum(dZ1, axis=0)
    
    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2
    return params

def evaluate_dataset(X, y, params):
    cache, loss = forward_pass(X, y, params)
    preds = np.argmax(cache["Y_prob"], axis=1)
    acc = np.mean(preds == y)
    return loss, acc, cache["Y_prob"], preds

def train_classifier(train_X, train_y, val_X, val_y, hidden_size, lr, num_epochs, batch_size, seed=7):
    input_dim = train_X.shape[1]
    output_dim = len(np.unique(train_y))
    params = init_params(input_dim, hidden_size, output_dim, seed)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(num_epochs):
        for b_idx in iterate_minibatches(len(train_X), batch_size, seed + epoch):
            b_X = train_X[b_idx]
            b_y = train_y[b_idx]
            cache, _ = forward_pass(b_X, b_y, params)
            params = backward_and_update(cache, params, lr)
            
        t_loss, t_acc, _, _ = evaluate_dataset(train_X, train_y, params)
        v_loss, v_acc, _, _ = evaluate_dataset(val_X, val_y, params)
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
    return params, history

def draw_sample_grid(fig, spec, images, labels, label_names):
    sample_indices = []
    for class_id in range(len(label_names)):     
        class_positions = np.where(labels == class_id)[0][:4]
        sample_indices.extend(class_positions.tolist())
    sub = spec.subgridspec(3, 4, wspace=0.15, hspace=0.35)
    for plot_idx, data_idx in enumerate(sample_indices):
        ax = fig.add_subplot(sub[plot_idx // 4, plot_idx % 4])
        ax.imshow(images[data_idx], cmap="gray")
        ax.set_title(label_names[labels[data_idx]], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

def draw_loss_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_loss"]) + 1) 
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=2.0, label="训练损失")
    ax.plot(epochs, history["val_loss"], color="#c1121f", linewidth=2.0, label="验证损失")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("训练 / 验证损失曲线")
    ax.legend(loc="upper right")

def draw_accuracy_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], color="#15803d", linewidth=2.0, label="训练精度")
    ax.plot(epochs, history["val_acc"], color="#f59e0b", linewidth=2.0, label="验证精度")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("训练 / 验证准确率曲线")
    ax.legend(loc="lower right")

def main():
    hidden_size = 64
    lr = 0.05
    num_epochs = 25
    batch_size = 32
    train_samples_per_class = 180
    val_samples_per_class = 60
    seed = 7
    
    train_X, train_y, train_images, val_X, val_y, val_images, label_names = load_fashion_subset(
        root="./data",      
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        seed=seed,
    )
    params, history = train_classifier(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        hidden_size=hidden_size,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,       
        seed=seed,
    )
    train_loss, train_acc, _, _ = evaluate_dataset(train_X, train_y, params)
    val_loss, val_acc, _, _ = evaluate_dataset(val_X, val_y, params)
    
    print("------ 三分类 softmax + 交叉熵实验 ------")
    print("hidden_size:", hidden_size)
    print("learning rate:", lr)
    print("num_epochs:", num_epochs)
    print("batch_size:", batch_size)
    print("final train loss:", train_loss)
    print("final train acc:", train_acc)
    print("final val loss:", val_loss)
    print("final val acc:", val_acc)
    
    fig = plt.figure(figsize=(15, 5.8))  
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    fig.text(0.06, 0.96, "真实图像三分类：softmax、交叉熵与手写关键反向传播", fontsize=18, weight="bold")
    draw_sample_grid(fig, gs[0], train_images, train_y, label_names)
    ax_loss = fig.add_subplot(gs[1])
    draw_loss_panel(ax_loss, history)
    ax_acc = fig.add_subplot(gs[2])
    draw_accuracy_panel(ax_acc, history)
    plt.show()

if __name__ == "__main__":
    main()