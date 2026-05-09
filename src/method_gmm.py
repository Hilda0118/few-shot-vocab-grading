import numpy as np
from sklearn.mixture import GaussianMixture
from .config import CFG


class GMMMethod:
    name = "GMM"

    def __init__(self, seed=0):
        self.seed = seed
        self.classes = list(CFG.all_classes)
        self.class_models = {}

    def _process_features(self, X, is_fit=False):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base, is_fit=True)
        self.y_base = y_base

        # ★ 关键：用所有 base 数据估计全局方差，供 novel 类借用
        self.global_var = np.var(self.X_base_proc, axis=0) + 1e-2  # shape (DIM,)
        self.global_prec_chol = 1.0 / np.sqrt(self.global_var)

        self.base_class_models = {}
        for c in CFG.base_classes:
            mask = (y_base == c)
            Xc = self.X_base_proc[mask]
            gmm = GaussianMixture(
                n_components=1, covariance_type="diag",
                reg_covar=1e-2, random_state=self.seed
            )
            gmm.fit(Xc)
            self.base_class_models[c] = gmm

    def predict_base(self, X):
        X_proc = self._process_features(X)
        scores = np.column_stack([
            self.base_class_models[c].score_samples(X_proc)
            for c in CFG.base_classes
        ])
        idx = np.argmax(scores, axis=1)
        return np.array([CFG.base_classes[i] for i in idx])

    def _safe_fit(self, Xc, n_support):
        """
        方差借用（Variance Borrowing）：
        - shot 少时：方差 ≈ 全局方差（稳定，不崩溃）
        - shot 多时：方差逐渐过渡到自身估计（更准确）
        - alpha = n_support / 30，最大为 1.0
        """
        if len(Xc) < 2:
            noise = np.random.RandomState(self.seed).normal(0, 1e-4, Xc.shape)
            Xc = np.vstack([Xc, Xc + noise])

        gmm = GaussianMixture(
            n_components=1, covariance_type="diag",
            reg_covar=1e-2, random_state=self.seed
        )
        gmm.fit(Xc)

        # ★ 核心修复：方差插值
        alpha = min(n_support / 30.0, 1.0)  # shot=1→0.03, shot=5→0.17, shot=20→0.67
        blended_var = (1.0 - alpha) * self.global_var + alpha * gmm.covariances_[0]
        gmm.covariances_[0] = blended_var
        gmm.precisions_chol_ = np.array([1.0 / np.sqrt(blended_var)])

        return gmm

    def adapt(self, X_support, y_support):
        X_support_proc = self._process_features(X_support)
        self.class_models = {}

        for c in self.classes:
            mask_base = (self.y_base == c)
            mask_supp = (y_support == c)
            n_support = int(mask_supp.sum())

            parts = []
            if mask_base.sum() > 0:
                parts.append(self.X_base_proc[mask_base])
            if mask_supp.sum() > 0:
                parts.append(X_support_proc[mask_supp])
            if len(parts) == 0:
                parts.append(self.X_base_proc)

            Xc = np.vstack(parts)
            self.class_models[c] = self._safe_fit(Xc, n_support)

    def predict(self, X):
        X_proc = self._process_features(X)
        scores = np.column_stack([
            self.class_models[c].score_samples(X_proc)
            for c in self.classes
        ])
        idx = np.argmax(scores, axis=1)
        return np.array([self.classes[i] for i in idx])

    def predict_score(self, X):
        X_proc = self._process_features(X)
        log_scores = np.column_stack([
            self.class_models[c].score_samples(X_proc)
            for c in self.classes
        ])
        log_scores -= np.max(log_scores, axis=1, keepdims=True)
        e = np.exp(log_scores)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)