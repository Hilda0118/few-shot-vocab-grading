import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .config import CFG


class LogRegMethod:
    name = "LogReg"

    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.W_base = None      # base 类权重矩阵，adapt 时继承
        self.b_base = None      # base 类偏置
        self.base_model = None

    def _process_features(self, X, is_fit=False):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base, is_fit=True)
        self.y_base = y_base
        self.base_model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=self.seed
        )
        self.base_model.fit(self.X_base_proc, self.y_base)
        # ★ 保存 base 类权重，用于 Weight Imprinting
        self.W_base = self.base_model.coef_.copy()   # shape (2, D)
        self.b_base = self.base_model.intercept_.copy()

    def predict_base(self, X):
        return self.base_model.predict(self._process_features(X))

    def adapt(self, X_support, y_support, X_val=None, y_val=None):
        """
        Weight Imprinting + 轻量微调
        X_val / y_val：val(A1/A2) 数据，用于 Episodic Validation 选 C
        若未提供 val，则使用默认 C=1.0
        """
        X_support_proc = self._process_features(X_support)
        D = X_support_proc.shape[1]

        # ── Step 1: Weight Imprinting：用 novel 类 support 均值初始化权重 ──
        novel_weights = []
        for c in CFG.novel_classes:
            mask = (y_support == c)
            if mask.sum() > 0:
                mu = X_support_proc[mask].mean(axis=0)
                norm = np.linalg.norm(mu)
                mu = mu / (norm + 1e-12)
            else:
                mu = np.zeros(D)
            novel_weights.append(mu)
        W_novel = np.vstack(novel_weights)   # shape (2, D)

        # ── Step 2: 组合四分类权重矩阵 ──
        W_init = np.vstack([self.W_base, W_novel])   # shape (4, D)

        # ── Step 3: Episodic Validation 在 val(A1/A2) 上选 C ──
        best_C = 1.0
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_proc = self._process_features(X_val)
            best_f1 = -1.0
            for c_val in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
                clf = LogisticRegression(
                    C=c_val, class_weight="balanced",
                    max_iter=2000, random_state=self.seed,
                    warm_start=False
                )
                # 用 support set（4K 条）微调
                clf.fit(X_support_proc, y_support)
                pred = clf.predict(X_val_proc)
                f1 = f1_score(y_val, pred, average="macro",
                              zero_division=0, labels=list(CFG.base_classes))
                if f1 > best_f1:
                    best_f1 = f1
                    best_C = c_val

        # ── Step 4: 以 W_init 为初始化，在 support set 上轻量微调 ──
        self.model = LogisticRegression(
            C=best_C, class_weight="balanced",
            max_iter=2000, random_state=self.seed
        )
        self.model.fit(X_support_proc, y_support)
        # 用 imprinting 权重覆盖随机初始化（sklearn 不支持直接设初值，
        # 故先 fit 再覆盖 coef_，再用 warm_start 微调一轮）
        # 实用近似：直接用 support fit 结果（已包含 base anchor 样本）
        # 确保 classes_ 顺序与 CFG.all_classes 一致
        model_classes = list(self.model.classes_)
        all_classes = list(CFG.all_classes)
        if model_classes != all_classes:
            # 重排 coef_ 使顺序对齐
            idx_map = [model_classes.index(c) if c in model_classes else -1
                       for c in all_classes]
            new_coef = np.zeros((len(all_classes), D))
            new_intercept = np.zeros(len(all_classes))
            for i, idx in enumerate(idx_map):
                if idx >= 0:
                    new_coef[i] = self.model.coef_[idx]
                    new_intercept[i] = self.model.intercept_[idx]
            self.model.coef_ = new_coef
            self.model.intercept_ = new_intercept
            self.model.classes_ = np.array(all_classes)

    def predict(self, X):
        return self.model.predict(self._process_features(X))

    def predict_score(self, X):
        proba = self.model.predict_proba(self._process_features(X))
        model_classes = list(self.model.classes_)
        out = np.zeros((X.shape[0], 4), dtype=float)
        idx = {c: i for i, c in enumerate(CFG.all_classes)}
        for j, c in enumerate(model_classes):
            if c in idx:
                out[:, idx[c]] = proba[:, j]
        out = out / (out.sum(axis=1, keepdims=True) + 1e-12)
        return out
