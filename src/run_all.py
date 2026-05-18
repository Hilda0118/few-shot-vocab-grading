import os
import time
import numpy as np
import pandas as pd

from .config import CFG
from .io_data import load_dataset, build_csv_from_pdf
from .split_protocol import split_train_val_test, split_base_novel, sample_support
from .embedding import Embedder
from .method_kmeans import KMeansMethod
from .method_knn import KNNMethod
from .method_gmm import GMMMethod
from .method_anchor import FixedAnchorMethod
from .method_logreg import LogRegMethod
from .method_ensemble import EnsembleMethod
from .metrics import eval_classification
from .plotting import plot_main_curves, plot_speed, plot_base_vs_novel, plot_context_ablation  # ← 改动①：新增导入

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def set_seed(seed):
    np.random.seed(seed)


# ────────────────────────────────────────────────
# ZeroShotDict 方法（零样本基线）
# ────────────────────────────────────────────────
class ZeroShotDictMethod:
    name = "ZeroShotDict"

    def __init__(self, embedder):
        self.embedder = embedder
        self.classes  = list(CFG.all_classes)

    def fit_base(self, X_base, y_base):
        defs = {
            "A1": [
                ("apple",  "I eat an apple every day."),
                ("water",  "She drinks water after school."),
                ("book",   "He reads a book at night."),
                ("family", "My family has four people."),
                ("time",   "What time is it now?"),
                ("day",    "Today is a sunny day."),
                ("hello",  "She said hello to her friend."),
                ("good",   "This is a good idea."),
            ],
            "A2": [
                ("accident",    "There was a car accident on the road."),
                ("careful",     "Be careful when you cross the street."),
                ("describe",    "Can you describe what happened?"),
                ("environment", "We must protect the environment."),
                ("familiar",    "This place looks familiar to me."),
                ("healthy",     "Eating vegetables keeps you healthy."),
            ],
            "B1": [
                ("absolutely", "I absolutely agree with your point."),
                ("accurate",   "The report must be accurate and clear."),
                ("background", "She has a strong academic background."),
                ("calculate",  "We need to calculate the total cost."),
                ("decade",     "The city changed a lot in the last decade."),
                ("efficient",  "This method is more efficient than before."),
            ],
            "B2": [
                ("abstract",   "The paper presents an abstract theoretical framework."),
                ("acquire",    "It takes years to acquire academic proficiency."),
                ("adequate",   "The evidence was not adequate to support the hypothesis."),
                ("capability", "The system demonstrated remarkable analytical capability."),
                ("dynamic",    "The dynamic interaction between variables complicates analysis."),
                ("hypothesis", "The researcher proposed a hypothesis based on prior data."),
            ],
        }
        self.anchors = {}
        for c in self.classes:
            pairs     = defs[c]
            words     = [p[0] for p in pairs]
            sentences = [p[1] for p in pairs]
            vecs      = self.embedder.transform(words, sentences)
            self.anchors[c] = vecs.mean(axis=0)
            self.anchors[c] /= (np.linalg.norm(self.anchors[c]) + 1e-12)

    def adapt(self, X_support, y_support):
        pass

    def predict(self, X):
        A  = np.vstack([self.anchors[c] for c in self.classes])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        return np.array([self.classes[i] for i in np.argmax(Xn @ An.T, axis=1)])

    def predict_score(self, X):
        A  = np.vstack([self.anchors[c] for c in self.classes])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ An.T
        z   = sim - np.max(sim, axis=1, keepdims=True)
        e   = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)


# ────────────────────────────────────────────────
# 工具：O(n) 特征对齐
# ────────────────────────────────────────────────
def get_features_by_df(sub_df, full_df, X_all):
    key2idxs = {}
    for i, row in full_df.iterrows():
        key = (row["word"], row["label"])
        key2idxs.setdefault(key, []).append(i)

    idxs        = []
    used_counts = {}
    for _, row in sub_df.iterrows():
        key  = (row["word"], row["label"])
        cnt  = used_counts.get(key, 0)
        cands = key2idxs.get(key, [])
        idxs.append(cands[cnt] if cnt < len(cands) else (cands[-1] if cands else 0))
        used_counts[key] = cnt + 1

    return X_all[np.array(idxs, dtype=int)]


# ────────────────────────────────────────────────
# 计算 base-pretrain 指标（fit_base 后、adapt 前）
# 只在 test(A1/A2) 上评估，不随 K 变化
# ────────────────────────────────────────────────
def eval_base_pretrain(method, X_test_base, y_test_base):
    """返回 fit_base 阶段结束后在 test(A1/A2) 上的 Macro-F1 和 per-class F1。"""
    try:
        if hasattr(method, "predict_base"):
            pred = method.predict_base(X_test_base)
        else:
            pred = method.predict(X_test_base)
        mf1 = f1_score(y_test_base, pred, average="macro",
                       labels=list(CFG.base_classes), zero_division=0)
        pf1 = f1_score(y_test_base, pred, average=None,
                       labels=list(CFG.base_classes), zero_division=0)
        acc = accuracy_score(y_test_base, pred)
        return {
            "base_pretrain_f1":     float(mf1),
            "base_pretrain_acc":    float(acc),
            "base_pretrain_f1_A1":  float(pf1[0]),
            "base_pretrain_f1_A2":  float(pf1[1]),
        }
    except Exception:
        return {
            "base_pretrain_f1":    np.nan,
            "base_pretrain_acc":   np.nan,
            "base_pretrain_f1_A1": np.nan,
            "base_pretrain_f1_A2": np.nan,
        }


# ────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────
def main():
    os.makedirs("data",          exist_ok=True)
    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.pred_dir,    exist_ok=True)
    os.makedirs(CFG.fig_dir,     exist_ok=True)

    if not os.path.exists(CFG.csv_path):
        print("[INFO] CSV not found, building from PDF...")
        build_csv_from_pdf(CFG.pdf_path, CFG.csv_path)

    if not os.path.exists(CFG.sentences_csv_path):
        print(f"[ERROR] 找不到带例句的数据集: {CFG.sentences_csv_path}")
        print("请先运行: python generate_sentences.py")
        return

    df = load_dataset(CFG.sentences_csv_path)
    print("[DATA] size:", len(df))
    print(df["label"].value_counts())

    embedder = Embedder(mode=CFG.embedding_mode, dim=CFG.embedding_dim)

    # ── 改动②：提取两种条件的特征（原来只提取一次，现在提取两次）──
    print("[INFO] Extracting features: with_context ...")
    X_all_with = embedder.transform(df["word"].tolist(),
                                    df["sentence"].tolist(),
                                    use_context=True)
    print("[INFO] Extracting features: no_context ...")
    X_all_no   = embedder.transform(df["word"].tolist(),
                                    df["sentence"].tolist(),
                                    use_context=False)
    print(f"[INFO] Feature matrix shape: {X_all_with.shape}")
    # ─────────────────────────────────────────────────────────────────

    val_methods = {"LogReg", "Ensemble"}

    all_metrics_rows = []  # ← 改动③：用于收集两个 condition 的全部原始数据，供消融图使用

    # ── 改动④：外层加 condition 循环，原 seed 循环整体向右缩进一级 ──
    for condition, X_all in [("with_context", X_all_with),
                              ("no_context",   X_all_no)]:

        print(f"\n{'#'*55}\n[CONDITION] {condition}\n{'#'*55}")

        metrics_rows, speed_rows, pred_rows = [], [], []  # 每个 condition 独立收集

        for seed in CFG.seeds:
            set_seed(seed)
            print(f"\n{'='*55}\n[SEED {seed}]")

            train_df, val_df, test_df = split_train_val_test(df, seed)
            # val_df 已只含 A1/A2（split_protocol 中过滤）
            base_df, novel_df = split_base_novel(train_df)

            print(f"  train={len(train_df)}, val={len(val_df)}(base only), "
                  f"test={len(test_df)}")

            X_base = get_features_by_df(base_df,  df, X_all)
            X_val  = get_features_by_df(val_df,   df, X_all)
            X_test = get_features_by_df(test_df,  df, X_all)
            y_base = base_df["label"].values
            y_val  = val_df["label"].values
            y_test = test_df["label"].values

            # test 集中 base / novel 子集
            base_test_mask  = np.isin(y_test, CFG.base_classes)
            X_test_base     = X_test[base_test_mask]
            y_test_base     = y_test[base_test_mask]

            for shot in CFG.shots:
                print(f"\n  [K={shot}]")
                support_df = sample_support(base_df, novel_df, shot, seed)
                X_support  = get_features_by_df(support_df, df, X_all)
                y_support  = support_df["label"].values

                method_list = [
                    KMeansMethod(seed),
                    KNNMethod(),
                    LogRegMethod(seed),
                    GMMMethod(seed),
                    FixedAnchorMethod(alpha=5.0),
                    EnsembleMethod(seed),
                ]
                if CFG.enable_zeroshot:
                    method_list.append(ZeroShotDictMethod(embedder))

                for method in method_list:

                    # ── Step 1: fit_base ──
                    method.fit_base(X_base, y_base)

                    # ── Step 2: 记录 base-pretrain（fit_base 后、adapt 前）──
                    pretrain_metrics = eval_base_pretrain(method, X_test_base, y_test_base)

                    # ── Step 3: warm-up（消除 JIT 影响）──
                    try:
                        if method.name in val_methods:
                            method.adapt(X_support, y_support, X_val, y_val)
                        else:
                            method.adapt(X_support, y_support)
                    except Exception:
                        pass

                    # ── Step 4: 正式计时 adapt ──
                    t0 = time.perf_counter()
                    try:
                        if method.name in val_methods:
                            method.adapt(X_support, y_support, X_val, y_val)
                        else:
                            method.adapt(X_support, y_support)
                    except Exception as e:
                        print(f"    [ERROR] {method.name} adapt: {e}")
                        continue
                    t1 = time.perf_counter()

                    # ── Step 5: predict ──
                    y_pred, y_score = None, None
                    try:
                        p0      = time.perf_counter()
                        y_pred  = method.predict(X_test)
                        y_score = method.predict_score(X_test)
                        p1      = time.perf_counter()
                    except Exception as e:
                        print(f"    [ERROR] {method.name} predict: {e}")
                        continue

                    # ── Step 6: 三层评估 ──
                    if y_pred is not None and y_score is not None:
                        m = eval_classification(y_test, y_pred, y_score)

                        # 合并 pretrain 指标 + 速度指标
                        m.update(pretrain_metrics)
                        m.update({
                            "method":     method.name,
                            "shot":       shot,
                            "seed":       seed,
                            "train_time": max(t1 - t0, 1e-6),
                            "infer_time": p1 - p0,
                            "latency_ms": (p1 - p0) / len(X_test) * 1000.0,
                            "throughput": len(X_test) / max((p1 - p0), 1e-9),
                        })
                        m["condition"] = condition  # ← 改动⑤：记录当前 condition
                        metrics_rows.append(m)
                        all_metrics_rows.append(m)  # ← 改动⑤：同时追加到总列表

                        pred_rows.append(pd.DataFrame({
                            "word":   test_df["word"].values,
                            "true":   y_test,
                            "pred":   y_pred,
                            "method": method.name,
                            "shot":   shot,
                            "seed":   seed,
                        }))

                        speed_rows.append({
                            "method":     method.name,
                            "shot":       shot,
                            "seed":       seed,
                            "train_time": max(t1 - t0, 1e-6),
                            "infer_time": p1 - p0,
                            "latency_ms": (p1 - p0) / len(X_test) * 1000.0,
                            "throughput": len(X_test) / max((p1 - p0), 1e-9),
                        })

                        print(f"    {method.name:15s} | "
                              f"Gen={m['macro_f1']:.3f} | "
                              f"Novel={m['novel_macro_f1']:.3f} | "
                              f"Base-pre={pretrain_metrics['base_pretrain_f1']:.3f} | "
                              f"Base-aft={m['base_after_f1']:.3f}")

        # ── 每个 condition 结束后：汇总保存（文件名加 condition 后缀）──
        metrics_df = pd.DataFrame(metrics_rows)
        speed_df   = pd.DataFrame(speed_rows)

        if pred_rows:
            preds_df = pd.concat(pred_rows, ignore_index=True)
            for (m_name, s), sub in preds_df.groupby(["method", "shot"]):
                sub.to_csv(
                    os.path.join(CFG.pred_dir, f"{m_name}_{s}_{condition}.csv"),
                    index=False
                )

        # ── 汇总统计 ──
        agg_dict = {
            # Generalized
            "accuracy":           ["mean", "std"],
            "macro_f1":           ["mean", "std"],
            "qwk":                ["mean", "std"],
            "mAP":                ["mean", "std"],
            "f1_A1":              "mean",
            "f1_A2":              "mean",
            "f1_B1":              "mean",
            "f1_B2":              "mean",
            # Base-pretrain（fit_base 后、adapt 前）
            "base_pretrain_f1":     ["mean", "std"],
            "base_pretrain_acc":    ["mean", "std"],
            "base_pretrain_f1_A1":  "mean",
            "base_pretrain_f1_A2":  "mean",
            # Base-after（adapt 后）
            "base_after_f1":        ["mean", "std"],
            "base_after_acc":       ["mean", "std"],
            "base_after_f1_A1":     "mean",
            "base_after_f1_A2":     "mean",
            # Novel-only
            "novel_macro_f1":       ["mean", "std"],
            "novel_acc":            ["mean", "std"],
            "novel_f1_B1":          "mean",
            "novel_f1_B2":          "mean",
        }
        summary = metrics_df.groupby(
            ["method", "shot"], as_index=False
        ).agg(agg_dict)
        summary.columns = [
            "_".join([c for c in col if c]).strip("_")
            for col in summary.columns
        ]
        summary.to_csv(
            os.path.join(CFG.outputs_dir, f"metrics_by_method_{condition}.csv"),
            index=False
        )

        speed_summary = speed_df.groupby(
            ["method", "shot"], as_index=False
        ).mean(numeric_only=True)
        speed_summary.to_csv(
            os.path.join(CFG.outputs_dir, f"speed_by_method_{condition}.csv"),
            index=False
        )

        # ── 各 condition 单独出图（原有绘图逻辑不变）──
        plot_main_curves(metrics_df,  suffix=f"_{condition}")
        plot_speed(speed_summary,  suffix=f"_{condition}")
        plot_base_vs_novel(metrics_df,  suffix=f"_{condition}")

    # ── 改动⑥：两个 condition 都跑完后，画消融对比图 ──────────────
    all_metrics_df = pd.DataFrame(all_metrics_rows)
    plot_context_ablation(all_metrics_df)
    # ─────────────────────────────────────────────────────────────────

    print("\n[DONE] Results saved in outputs/")


if __name__ == "__main__":
    main()
