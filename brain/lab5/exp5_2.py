import numpy as np
import matplotlib
# 【核心修复1】强制设置Matplotlib后端，同时提前配置Tkinter字体
matplotlib.use('TkAgg')
import warnings
# 【核心修复2】屏蔽所有Tkinter字体缺失警告，彻底解决报错！
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
import tkinter.font as tkfont

# 【核心修复3】配置Tkinter全局默认中文字体（修复控件中文）
root = tk.Tk()
default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(family="Microsoft YaHei", size=10)
root.destroy()

# 【核心修复4】Matplotlib绘图字体（修复图表中文）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


# 根据赫布规则更新权重：相关输入越强，且能驱动输出发放，其权重越容易增强
def hebb_update(w, x, y, eta, w_min=0.0, w_max=1.0):
    delta = eta * x * y
    w_new = w + delta
    return np.clip(w_new, w_min, w_max)


# 根据输入发放概率生成一段二值脉冲序列，固定随机种子便于复现实验结果
def generate_spike_train(steps, prob, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(steps) < prob).astype(float)


# 用给定参数完整模拟一次双输入赫布学习过程
def simulate_hebb_process(steps, eta, threshold, prob_a, prob_b, decay=0.85, seed=7):
    x_a = generate_spike_train(steps, prob_a, seed=seed)
    x_b = generate_spike_train(steps, prob_b, seed=seed + 1)

    w = np.array([0.10, 0.10], dtype=float)
    membrane = 0.0
    outputs = []
    w_a_history = [w[0]]
    w_b_history = [w[1]]

    for t in range(steps):
        x = np.array([x_a[t], x_b[t]])
        syn_input = float(np.dot(w, x))
        membrane = decay * membrane + syn_input

        if membrane >= threshold:
            y = 1.0
            membrane = 0.0
        else:
            y = 0.0

        w = hebb_update(w, x, y, eta)

        outputs.append(y)
        w_a_history.append(w[0])
        w_b_history.append(w[1])

    return {
        "x_a": x_a,
        "x_b": x_b,
        "y": np.array(outputs),
        "w_a_history": np.array(w_a_history),
        "w_b_history": np.array(w_b_history),
        "w_final": w,
        "total_output_spikes": int(np.sum(outputs)),
    }


# 在一张坐标轴中画出输入 A、输入 B、输出 y 的发放时刻
def draw_spike_panel(ax, sim_data, steps):
    ax.clear()
    x_a_times = np.where(sim_data["x_a"] > 0)[0]
    x_b_times = np.where(sim_data["x_b"] > 0)[0]
    y_times = np.where(sim_data["y"] > 0)[0]

    ax.eventplot(
        [x_a_times, x_b_times, y_times],
        lineoffsets=[2, 1, 0],
        linelengths=0.6,
        colors=["#2563eb", "#15803d", "#555555"],
        linewidths=2.0,
    )
    ax.set_xlim(-0.5, steps - 0.5)
    ax.set_ylim(-0.8, 2.8)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["输入 A", "输入 B", "输出 y"])
    ax.set_xticks([])
    ax.set_title("输入与输出脉冲时刻")
    ax.grid(axis="x", alpha=0.15)


# 在另一张坐标轴中画出 wA、wB 的权重演化曲线
def draw_weight_panel(ax, sim_data, steps):
    ax.clear()
    time = np.arange(steps + 1)
    ax.plot(time, sim_data["w_a_history"], marker="o", markersize=4, color="#2563eb", label="权重 wA")
    ax.plot(time, sim_data["w_b_history"], marker="o", markersize=4, color="#15803d", label="权重 wB")
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("时间步")
    ax.set_ylabel("权重值")
    ax.set_title("权重演化曲线")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")


def main():
    steps = 40
    eta_init = 0.05
    threshold_init = 0.5
    prob_a_init = 0.80
    prob_b_init = 0.45

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    plt.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.22, hspace=0.35)

    ax_spikes = fig.add_subplot(gs[0])
    ax_weights = fig.add_subplot(gs[1])

    title_text = fig.text(0.08, 0.92, "赫布学习模型：双输入权重演化", fontsize=18, weight="bold", color="#333333")
    stats_text = fig.text(0.72, 0.90, "", fontsize=14, family="monospace", color="#333333")

    sim_data = simulate_hebb_process(
        steps=steps,
        eta=eta_init,
        threshold=threshold_init,
        prob_a=prob_a_init,
        prob_b=prob_b_init,
    )
    draw_spike_panel(ax_spikes, sim_data, steps)
    draw_weight_panel(ax_weights, sim_data, steps)
    stats_text.set_text(
        f"终值 wA  {sim_data['w_final'][0]:.3f}    终值 wB  {sim_data['w_final'][1]:.3f}    总输出脉冲  {sim_data['total_output_spikes']}"
    )

    slider_y1 = 0.12
    slider_y2 = 0.06
    ax_eta = fig.add_axes([0.33, slider_y1, 0.12, 0.03], facecolor="#f5f5f5")
    ax_threshold = fig.add_axes([0.79, slider_y1, 0.12, 0.03], facecolor="#f5f5f5")
    ax_prob_a = fig.add_axes([0.33, slider_y2, 0.12, 0.03], facecolor="#f5f5f5")
    ax_prob_b = fig.add_axes([0.79, slider_y2, 0.12, 0.03], facecolor="#f5f5f5")

    fig.text(0.08, slider_y1 + 0.01, "学习率 (eta)", fontsize=14)
    fig.text(0.56, slider_y1 + 0.01, "激活阈值 (Threshold)", fontsize=14)
    fig.text(0.08, slider_y2 + 0.01, "输入A 发放概率 (ProbA)", fontsize=14)
    fig.text(0.56, slider_y2 + 0.02, "输入B 发放概率 (ProbB)", fontsize=14)

    slider_eta = Slider(ax=ax_eta, label="", valmin=0.0, valmax=0.20, valinit=eta_init, valstep=0.01, color="#2563eb")
    slider_threshold = Slider(ax=ax_threshold, label="", valmin=0.05, valmax=1.5, valinit=threshold_init, valstep=0.05, color="#2563eb")
    slider_prob_a = Slider(ax=ax_prob_a, label="", valmin=0.0, valmax=1.0, valinit=prob_a_init, valstep=0.05, color="#2563eb")
    slider_prob_b = Slider(ax=ax_prob_b, label="", valmin=0.0, valmax=1.0, valinit=prob_b_init, valstep=0.05, color="#2563eb")

    def update(_):
        sim_data = simulate_hebb_process(
            steps=steps,
            eta=float(slider_eta.val),
            threshold=float(slider_threshold.val),
            prob_a=float(slider_prob_a.val),
            prob_b=float(slider_prob_b.val),
        )

        draw_spike_panel(ax_spikes, sim_data, steps)
        draw_weight_panel(ax_weights, sim_data, steps)
        stats_text.set_text(
            f"终值 wA  {sim_data['w_final'][0]:.3f}    终值 wB  {sim_data['w_final'][1]:.3f}    总输出脉冲  {sim_data['total_output_spikes']}"
        )
        fig.canvas.draw_idle()

    slider_eta.on_changed(update)
    slider_threshold.on_changed(update)
    slider_prob_a.on_changed(update)
    slider_prob_b.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()