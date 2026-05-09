import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CFG


def plot_main_curves(metrics_df):
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid")

    df = metrics_df.copy()
    df["shot_str"] = df["shot"].astype(str)

    # Macro-F1
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=df, x="shot_str", y="macro_f1",
        hue="method", marker="o", estimator=np.mean, errorbar="sd"
    )
    plt.title("Macro-F1 vs Shot")
    ax.legend(title="method", loc="upper left", ncol=2, frameon=True)
    plt.savefig(os.path.join(CFG.fig_dir, "fig_macro_f1.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # QWK
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=df, x="shot_str", y="qwk",
        hue="method", marker="o", estimator=np.mean, errorbar="sd"
    )
    plt.title("QWK vs Shot")
    ax.legend(title="method", loc="upper left", ncol=2, frameon=True)
    plt.savefig(os.path.join(CFG.fig_dir, "fig_qwk.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # mAP
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=df, x="shot_str", y="mAP",
        hue="method", marker="o", estimator=np.mean, errorbar="sd"
    )
    plt.title("mAP vs Shot")
    ax.legend(title="method", loc="upper left", ncol=2, frameon=True)
    plt.savefig(os.path.join(CFG.fig_dir, "fig_map.png"), dpi=220, bbox_inches="tight")
    plt.close()


def plot_speed(speed_df):
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Train time
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=speed_df, x="method", y="train_time", hue="shot",
        errorbar=None   # 替代旧的 ci=None
    )
    plt.title("Train Time by Method/Shot")
    ax.legend(title="shot", loc="upper left", ncol=2, frameon=True)
    plt.savefig(os.path.join(CFG.fig_dir, "fig_train_time.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Throughput
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=speed_df, x="method", y="throughput", hue="shot",
        errorbar=None
    )
    plt.title("Throughput by Method/Shot")
    ax.legend(title="shot", loc="upper left", ncol=2, frameon=True)
    plt.savefig(os.path.join(CFG.fig_dir, "fig_throughput.png"), dpi=220, bbox_inches="tight")
    plt.close()


def plot_forgetting_heatmap(metrics_df):
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid")

    pv = metrics_df.groupby(["method", "shot"])["forgetting_rate"].mean().reset_index()
    table = pv.pivot(index="method", columns="shot", values="forgetting_rate").fillna(0)

    plt.figure(figsize=(8, 5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Forgetting Rate Heatmap")
    plt.savefig(os.path.join(CFG.fig_dir, "fig_forgetting_heatmap.png"), dpi=220, bbox_inches="tight")
    plt.close()
