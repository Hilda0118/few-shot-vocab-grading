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
from .plotting import plot_main_curves, plot_speed, plot_forgetting_heatmap

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def set_seed(seed):
    np.random.seed(seed)

class ZeroShotDictMethod:
    name = "ZeroShotDict"

    def __init__(self, embedder):
        self.embedder = embedder
        self.classes = list(CFG.all_classes)

    def fit_base(self, X_base, y_base):
        # 【修改】每个 anchor 词配上对应级别的例句，
        # 确保 anchor 向量和训练数据在同一个特征空间（带语境的 768 维空间）
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
                ("accident",     "There was a car accident on the road."),
                ("careful",      "Be careful when you cross the street."),
                ("describe",     "Can you describe what happened?"),
                ("environment",  "We must protect the environment."),
                ("familiar",     "This place looks familiar to me."),
                ("healthy",      "Eating vegetables keeps you healthy."),
            ],
            "B1": [
                ("absolutely",  "I absolutely agree with your point."),
                ("accurate",    "The report must be accurate and clear."),
                ("background",  "She has a strong academic background."),
                ("calculate",   "We need to calculate the total cost."),
                ("decade",      "The city changed a lot in the last decade."),
                ("efficient",   "This method is more efficient than before."),
            ],
            "B2": [
                ("abstract",    "The paper presents an abstract theoretical framework."),
                ("acquire",     "It takes years to acquire academic proficiency."),
                ("adequate",    "The evidence was not adequate to support the hypothesis."),
                ("capability",  "The system demonstrated remarkable analytical capability."),
                ("dynamic",     "The dynamic interaction between variables complicates analysis."),
                ("hypothesis",  "The researcher proposed a hypothesis based on prior data."),
            ],
        }
        self.anchors = {}
        for c in self.classes:
            pairs = defs[c]
            words     = [p[0] for p in pairs]
            sentences = [p[1] for p in pairs]
            vecs = self.embedder.transform(words, sentences)  # 【修改】传入 sentences
            self.anchors[c] = vecs.mean(axis=0)
            self.anchors[c] = self.anchors[c] / (np.linalg.norm(self.anchors[c]) + 1e-12)

    def adapt(self, X_support, y_support):
        pass

    def predict(self, X):
        A = np.vstack([self.anchors[c] for c in self.classes])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ An.T
        idx = np.argmax(sim, axis=1)
        return np.array([self.classes[i] for i in idx])

    def predict_score(self, X):
        A = np.vstack([self.anchors[c] for c in self.classes])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ An.T
        z = sim - np.max(sim, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.pred_dir, exist_ok=True)
    os.makedirs(CFG.fig_dir, exist_ok=True)

    if not os.path.exists(CFG.csv_path):
        print("[INFO] CSV not found, building from PDF...")
        build_csv_from_pdf(CFG.pdf_path, CFG.csv_path)

    if not os.path.exists(CFG.sentences_csv_path):
        print(f"[ERROR] 找不到带例句的数据集: {CFG.sentences_csv_path}")
        print("请先在终端运行: python generate_sentences.py")
        return

    df = load_dataset(CFG.sentences_csv_path)
    print("[DATA] size:", len(df))
    print(df["label"].value_counts())

    embedder = Embedder(mode=CFG.embedding_mode, dim=CFG.embedding_dim)

    X_all = embedder.transform(df["word"].tolist(), df["sentence"].tolist())
    y_all = df["label"].values

    metrics_rows, speed_rows, pred_rows = [], [], []

    for seed in CFG.seeds:
        set_seed(seed)
        train_df, val_df, test_df = split_train_val_test(df, seed)
        base_df, novel_df = split_base_novel(train_df)

        def to_idx(sub_df):
            idxs = []
            used = set()
            for w, l in zip(sub_df["word"], sub_df["label"]):
                found = None
                for i, (ww, ll) in enumerate(zip(df["word"], df["label"])):
                    if i in used:
                        continue
                    if ww == w and ll == l:
                        found = i
                        used.add(i)
                        break
                if found is None:
                    found = df[(df["word"] == w) & (df["label"] == l)].index[0]
                idxs.append(found)
            return np.array(idxs, dtype=int)

        idx_base  = to_idx(base_df)
        idx_val   = to_idx(val_df)
        idx_test  = to_idx(test_df)

        X_base, y_base = X_all[idx_base], y_all[idx_base]
        X_val,  y_val  = X_all[idx_val],  y_all[idx_val]
        X_test, y_test = X_all[idx_test], y_all[idx_test]

        base_test_mask = np.isin(y_test, CFG.base_classes)
        X_test_base, y_test_base = X_test[base_test_mask], y_test[base_test_mask]

        for shot in CFG.shots:
            support_df = sample_support(base_df, novel_df, shot, seed)
            idx_support = to_idx(support_df)
            X_support, y_support = X_all[idx_support], y_all[idx_support]

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
                method.fit_base(X_base, y_base)

                before = np.nan
                try:
                    from sklearn.metrics import accuracy_score
                    if hasattr(method, "predict_base"):
                        pred_before = method.predict_base(X_test_base)
                    else:
                        pred_before = method.predict(X_test_base)
                    before = accuracy_score(y_test_base, pred_before)
                except Exception:
                    pass

                # 【修改】去掉 "KNN" 和 "SVM"（这两个 name 已不存在），
                # 只保留实际存在的 name：ProtoNet / LogReg / Ensemble
                val_methods = {"ProtoNet", "LogReg", "Ensemble"}

                # -------- warm-up --------
                try:
                    if method.name in val_methods:
                        method.adapt(X_support, y_support, X_val, y_val)
                    else:
                        method.adapt(X_support, y_support)
                except Exception:
                    pass

                # -------- 正式计时 --------
                t0 = time.perf_counter()
                try:
                    if method.name in val_methods:
                        method.adapt(X_support, y_support, X_val, y_val)
                    else:
                        method.adapt(X_support, y_support)
                except Exception as e:
                    print(f"[ERROR] {method.name} adapt 阶段报错: {e}")
                    continue
                t1 = time.perf_counter()

                y_pred, y_score = None, None
                try:
                    p0 = time.perf_counter()
                    y_pred  = method.predict(X_test)
                    y_score = method.predict_score(X_test)
                    p1 = time.perf_counter()
                except Exception as e:
                    print(f"[ERROR] {method.name} predict 阶段报错: {e}")
                    continue

                after = np.nan
                try:
                    from sklearn.metrics import accuracy_score
                    if hasattr(method, "predict_base"):
                        pred_after = method.predict_base(X_test_base)
                    else:
                        pred_after = method.predict(X_test_base)
                    after = accuracy_score(y_test_base, pred_after)
                except Exception:
                    pass

                forgetting = np.nan
                if not (np.isnan(before) or np.isnan(after)):
                    forgetting = max(0.0, before - after)

                if y_pred is not None and y_score is not None:
                    m = eval_classification(y_test, y_pred, y_score)
                    m.update({
                        "method":          method.name,
                        "shot":            shot,
                        "seed":            seed,
                        "train_time":      max(t1 - t0, 1e-6),
                        "infer_time":      p1 - p0,
                        "latency_ms":      (p1 - p0) / len(X_test) * 1000.0,
                        "throughput":      len(X_test) / max((p1 - p0), 1e-9),
                        "forgetting_rate": forgetting,
                    })
                    metrics_rows.append(m)

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

    metrics_df = pd.DataFrame(metrics_rows)
    speed_df   = pd.DataFrame(speed_rows)
    preds_df   = pd.concat(pred_rows, ignore_index=True)

    for (m, s), sub in preds_df.groupby(["method", "shot"]):
        sub.to_csv(os.path.join(CFG.pred_dir, f"{m}_{s}.csv"), index=False)

    summary = metrics_df.groupby(["method", "shot"], as_index=False).agg({
        "accuracy":        ["mean", "std"],
        "macro_f1":        ["mean", "std"],
        "qwk":             ["mean", "std"],
        "mAP":             ["mean", "std"],
        "f1_A1":           "mean",
        "f1_A2":           "mean",
        "f1_B1":           "mean",
        "f1_B2":           "mean",
        "forgetting_rate": ["mean", "std"],
    })
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns]
    summary.to_csv(os.path.join(CFG.outputs_dir, "metrics_by_method.csv"), index=False)

    speed_summary = speed_df.groupby(["method", "shot"], as_index=False).mean(numeric_only=True)
    speed_summary.to_csv(os.path.join(CFG.outputs_dir, "speed_by_method.csv"), index=False)

    stability = metrics_df.groupby(["method", "shot"], as_index=False).agg({
        "macro_f1": ["mean", "std"],
        "qwk":      ["mean", "std"],
        "mAP":      ["mean", "std"],
    })
    stability.columns = ["_".join([c for c in col if c]).strip("_") for col in stability.columns]
    stability.to_csv(os.path.join(CFG.outputs_dir, "stability_by_method.csv"), index=False)

    plot_main_curves(metrics_df)
    plot_speed(speed_summary)
    plot_forgetting_heatmap(metrics_df)

    print("[DONE] Results saved in outputs/")

if __name__ == "__main__":
    main()