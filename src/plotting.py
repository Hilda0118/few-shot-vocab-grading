import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CFG

METHOD_ORDER = ["LogReg", "GMM", "ProtoNet", "FixedAnchor",
                "KMeans", "Ensemble", "ZeroShotDict"]
PALETTE      = "tab10"
_COLORS      = {m: c for m, c in zip(
    METHOD_ORDER,
    plt.cm.tab10(np.linspace(0, 1, 10))[:len(METHOD_ORDER)]
)}

_AGG_COLS = [
    "accuracy", "macro_f1", "qwk", "mAP",
    "f1_A1", "f1_A2", "f1_B1", "f1_B2",
    "base_pretrain_f1", "base_pretrain_acc",
    "base_pretrain_f1_A1", "base_pretrain_f1_A2",
    "base_after_f1", "base_after_acc",
    "base_after_f1_A1", "base_after_f1_A2",
    "novel_macro_f1", "novel_acc",
    "novel_f1_B1", "novel_f1_B2",
]


def _aggregate(df):
    if any(c.endswith("_mean") for c in df.columns):
        df = df.copy()
        df.columns = [c[:-5] if c.endswith("_mean") else c for c in df.columns]
        df = df[[c for c in df.columns if not c.endswith("_std")]]
        return df.sort_values(["method", "shot"]).reset_index(drop=True)

    if df.groupby(["method", "shot"]).size().max() == 1:
        return df.sort_values(["method", "shot"]).reset_index(drop=True)

    num_cols = [c for c in _AGG_COLS if c in df.columns]
    df_agg = (df.groupby(["method", "shot"], as_index=False)[num_cols]
                .mean()
                .sort_values(["method", "shot"])
                .reset_index(drop=True))
    return df_agg


def _col(df, name):
    if name in df.columns:
        return name
    if name + "_mean" in df.columns:
        return name + "_mean"
    raise KeyError(f"plotting: 找不到列 '{name}'，实际列: {list(df.columns)}")


def _prepare(df):
    df = df.copy().sort_values("shot")
    df["shot_str"] = df["shot"].astype(str)
    return df


def _lineplot_ax(ax, data, x, y_col):
    methods = [m for m in METHOD_ORDER if m in data["method"].unique()]
    for method in methods:
        sub = data[data["method"] == method].sort_values("shot")
        ax.plot(sub[x], sub[y_col],
                color=_COLORS[method], marker="o",
                linewidth=2, label=method)


def _facet_setup(n_methods, sharey=False):
    ncols = 4
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 3.2),
                             sharey=sharey)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    return fig, axes_flat, nrows, ncols


# ──────────────────────────────────────────────────────
# 图一：Generalized 层 — 三子图竖排
# ──────────────────────────────────────────────────────
def plot_main_curves(metrics_df, suffix=""):          # ← 改动：加 suffix=""
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.05)
    df = _prepare(_aggregate(metrics_df))

    fig, axes = plt.subplots(3, 1, figsize=(9, 14))
    fig.suptitle("Generalized Evaluation (A1/A2/B1/B2, after adapt)",
                 fontsize=14, fontweight="bold")

    configs = [
        ("macro_f1", "Macro-F1",  "① Macro-F1 vs K"),
        ("qwk",      "QWK",       "② QWK vs K"),
        ("mAP",      "mAP",       "③ mAP vs K"),
    ]
    for ax, (col, ylabel, title) in zip(axes, configs):
        _lineplot_ax(ax, df, "shot_str", _col(df, col))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("K (shots per class)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(title="method", fontsize=8, title_fontsize=9,
                  loc="upper left", ncol=1, frameon=True)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_generalized{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────
# 图二：Base pretrain vs after — 分面
# ──────────────────────────────────────────────────────
def _plot_base_comparison(metrics_df, suffix=""):     # ← 改动：加 suffix=""
    os.makedirs(CFG.fig_dir, exist_ok=True)
    df = _prepare(_aggregate(metrics_df))
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    col_pre = _col(df, "base_pretrain_f1")
    col_aft = _col(df, "base_after_f1")

    fig, axes_flat, nrows, ncols = _facet_setup(len(methods))

    for i, method in enumerate(methods):
        ax    = axes_flat[i]
        sub   = df[df["method"] == method].sort_values("shot")
        color = _COLORS[method]
        shots = sub["shot_str"].tolist()

        pretrain_val = sub[col_pre].iloc[0]
        after_vals   = sub[col_aft].values

        ax.axhline(pretrain_val, color=color, linestyle="--",
                   linewidth=2, label="pretrain")
        ax.plot(shots, after_vals, color=color, linestyle="-",
                marker="o", linewidth=2, label="after")
        ax.fill_between(shots, [pretrain_val]*len(shots), after_vals,
                        color=color, alpha=0.12)

        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("K", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Base Macro-F1", fontsize=9)
        ax.legend(fontsize=8, loc="lower right", frameon=True)
        ax.grid(axis="y", alpha=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Base Macro-F1: Pretrain vs After Adapt",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_base_comparison{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────
# 图三：遗忘量 ΔF1
# ──────────────────────────────────────────────────────
def _plot_forgetting(metrics_df, suffix=""):          # ← 改动：加 suffix=""
    os.makedirs(CFG.fig_dir, exist_ok=True)
    df = _prepare(_aggregate(metrics_df))
    col_pre = _col(df, "base_pretrain_f1")
    col_aft = _col(df, "base_after_f1")
    df["forgetting"] = df[col_pre] - df[col_aft]
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df[df["method"] == method].sort_values("shot")
        ax.plot(sub["shot_str"], sub["forgetting"],
                color=_COLORS[method], marker="o",
                linewidth=2, label=method)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Base Class Forgetting  ΔF1 = pretrain − after", fontsize=12)
    ax.set_xlabel("K (shots per class)", fontsize=10)
    ax.set_ylabel("ΔF1", fontsize=10)
    ax.legend(title="method", fontsize=8, title_fontsize=9,
              loc="upper right", ncol=1, frameon=True)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_forgetting{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────
# 图四：Novel vs Base-After — 分面
# ──────────────────────────────────────────────────────
def _plot_novel_vs_base_after(metrics_df, suffix=""):  # ← 改动：加 suffix=""
    os.makedirs(CFG.fig_dir, exist_ok=True)
    df = _prepare(_aggregate(metrics_df))
    col_aft   = _col(df, "base_after_f1")
    col_novel = _col(df, "novel_macro_f1")
    methods   = [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, axes_flat, nrows, ncols = _facet_setup(len(methods))

    for i, method in enumerate(methods):
        ax    = axes_flat[i]
        sub   = df[df["method"] == method].sort_values("shot")
        color = _COLORS[method]
        shots = sub["shot_str"].tolist()

        base_after = sub[col_aft].values
        novel      = sub[col_novel].values

        ax.plot(shots, base_after, color=color, linestyle="--",
                marker="s", linewidth=2, label="base-after")
        ax.plot(shots, novel, color=color, linestyle="-",
                marker="o", linewidth=2, label="novel")

        for s, nv, ba in zip(shots, novel, base_after):
            sym   = "▲" if nv >= ba else "▼"
            clr   = "green" if nv >= ba else "red"
            ypos  = nv + 0.04 if nv >= ba else nv - 0.05
            ax.text(s, ypos, sym, ha="center", color=clr, fontsize=9)

        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("K", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Macro-F1", fontsize=9)
        ax.legend(fontsize=8, loc="upper left", frameon=True)
        ax.grid(axis="y", alpha=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Novel vs Base-After Macro-F1  (▲ novel wins  ▼ base wins)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_novel_vs_base_after{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df[df["method"] == method].sort_values("shot")
        ax2.plot(sub["shot_str"], sub[col_novel],
                 color=_COLORS[method], marker="o",
                 linewidth=2, label=method)
    ax2.set_title("Novel-only Macro-F1 vs K", fontsize=12)
    ax2.set_xlabel("K (shots per class)", fontsize=10)
    ax2.set_ylabel("Novel Macro-F1", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.legend(title="method", fontsize=8, title_fontsize=9,
               loc="upper left", ncol=1, frameon=True)
    ax2.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_novel{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────
# 图五：雷达图
# ── 改动：加 suffix + 修复 float() 防止类型报错 ─────────
# ──────────────────────────────────────────────────────
def _plot_radar(metrics_df, suffix=""):               # ← 改动：加 suffix=""
    df_agg = _aggregate(metrics_df)
    df_k   = df_agg[df_agg["shot"] == df_agg["shot"].max()].copy()
    if df_k.empty:
        return

    methods    = [m for m in METHOD_ORDER if m in df_k["method"].unique()]
    categories = ["F1(A1)", "F1(A2)", "F1(B1)", "F1(B2)"]
    col_map    = {
        "F1(A1)": _col(df_k, "f1_A1"),
        "F1(A2)": _col(df_k, "f1_A2"),
        "F1(B1)": _col(df_k, "f1_B1"),
        "F1(B2)": _col(df_k, "f1_B2"),
    }
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method in methods:
        sub    = df_k[df_k["method"] == method]
        values = [float(sub[col_map[c]].mean()) for c in categories]  # ← 改动：加 float()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=method, color=_COLORS[method])
        ax.fill(angles, values, alpha=0.08, color=_COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class F1 @ K={int(df_agg['shot'].max())}",
                 size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, f"fig_radar{suffix}.png"),  # ← 改动
                dpi=220, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────
# 速度图
# ──────────────────────────────────────────────────────
def plot_speed(speed_df, suffix=""):                  # ← 改动：加 suffix=""
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.05)

    for col, ylabel, fname_base, palette in [
        ("train_time",  "Time (s)",    "fig_train_time",  "Blues_d"),   # ← 改动：去掉 .png
        ("throughput",  "Samples / s", "fig_throughput",  "Greens_d"),  # ← 改动：去掉 .png
    ]:
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=speed_df, x="method", y=col,
                         hue="shot", errorbar=None, palette=palette)
        ax.set_title(
            f"{'Adapt Time' if col == 'train_time' else 'Throughput'} by Method / Shot")
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.legend(title="shot", loc="upper right", ncol=2, frameon=True)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.fig_dir, f"{fname_base}{suffix}.png"),  # ← 改动
                    dpi=220, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────
def plot_base_vs_novel(metrics_df, suffix=""):        # ← 改动：加 suffix=""
    _plot_base_comparison(metrics_df, suffix)         # ← 改动：透传
    _plot_forgetting(metrics_df, suffix)              # ← 改动：透传
    _plot_novel_vs_base_after(metrics_df, suffix)     # ← 改动：透传
    _plot_radar(metrics_df, suffix)                   # ← 改动：透传


# ──────────────────────────────────────────────────────
# 消融对比：有语境 vs 无语境 — 7方法分面
# ──────────────────────────────────────────────────────
def plot_context_ablation(metrics_df):
    os.makedirs(CFG.fig_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.05)

    num_cols = [c for c in _AGG_COLS if c in metrics_df.columns]
    df = (metrics_df
          .groupby(["method", "shot", "condition"], as_index=False)[num_cols]
          .mean()
          .sort_values(["method", "shot"])
          .reset_index(drop=True))
    df["shot_str"] = df["shot"].astype(str)

    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    fig, axes_flat, nrows, ncols = _facet_setup(len(methods))

    for i, method in enumerate(methods):
        ax    = axes_flat[i]
        color = _COLORS[method]

        for condition, linestyle, marker, label_suffix in [
            ("with_context", "-",  "o", "with context"),
            ("no_context",   "--", "s", "without context"),
        ]:
            sub = df[(df["method"] == method) &
                     (df["condition"] == condition)].sort_values("shot")
            if sub.empty:
                continue
            ax.plot(sub["shot_str"], sub["macro_f1"],
                    color=color, linestyle=linestyle, marker=marker,
                    linewidth=2, label=label_suffix)

        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("K", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Generalized Macro-F1", fontsize=9)
        ax.legend(fontsize=8, loc="upper left", frameon=True)
        ax.grid(axis="y", alpha=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Context Ablation: Generalized Macro-F1\n"
                 "(solid — with context    dashed - - without context)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.fig_dir, "fig_context_ablation.png"),
                dpi=220, bbox_inches="tight")
    plt.close()
    print("[PLOT] fig_context_ablation.png saved.")
