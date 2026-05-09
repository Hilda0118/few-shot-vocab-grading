import numpy as np
from .config import CFG


class KNNMethod:
    name = "ProtoNet"

    def __init__(self):
        self.model = None
        self.classes = list(CFG.all_classes)

    def _process_features(self, X):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base)
        self.y_base = y_base
        self.base_prototypes = {}
        # 【新增】记录全局均值，用于低shot时的平滑
        self.global_mean = self.X_base_proc.mean(axis=0)
        self.global_mean /= (np.linalg.norm(self.global_mean) + 1e-10)

        for c in CFG.base_classes:
            mask = (y_base == c)
            if mask.sum() > 0:
                p = self.X_base_proc[mask].mean(axis=0)
                self.base_prototypes[c] = p / (np.linalg.norm(p) + 1e-10)

    def predict_base(self, X):
        X_proc = self._process_features(X)
        protos = np.vstack([self.base_prototypes[c] for c in CFG.base_classes])
        dist = np.linalg.norm(X_proc[:, np.newaxis, :] - protos[np.newaxis, :, :], axis=2)
        idx = np.argmin(dist, axis=1)
        return np.array([CFG.base_classes[i] for i in idx])

    def adapt(self, X_support, y_support, X_val=None, y_val=None):
        X_support_proc = self._process_features(X_support)
        self.prototypes = {}

        for c in CFG.base_classes:
            self.prototypes[c] = self.base_prototypes[c]

        for c in CFG.novel_classes:
            mask = (y_support == c)
            k = mask.sum()
            if k > 0:
                p_supp = X_support_proc[mask].mean(axis=0)
                p_supp /= (np.linalg.norm(p_supp) + 1e-10)
                # 【修改】低shot时向全局均值平滑，shot越多越信任自己的prototype
                # k=1时 lam=0.17, k=5时 lam=0.5, k=20时 lam=0.8
                lam = k / (k + 5.0)
                p = lam * p_supp + (1.0 - lam) * self.global_mean
            else:
                p = self.global_mean.copy()
            self.prototypes[c] = p / (np.linalg.norm(p) + 1e-10)

    def predict(self, X):
        X_proc = self._process_features(X)
        protos = np.vstack([self.prototypes[c] for c in self.classes])
        dist = np.linalg.norm(X_proc[:, np.newaxis, :] - protos[np.newaxis, :, :], axis=2)
        idx = np.argmin(dist, axis=1)
        return np.array([self.classes[i] for i in idx])

    def predict_score(self, X):
        X_proc = self._process_features(X)
        protos = np.vstack([self.prototypes[c] for c in self.classes])
        dist_sq = np.sum((X_proc[:, np.newaxis, :] - protos[np.newaxis, :, :]) ** 2, axis=2)
        temperature = np.mean(dist_sq) / 2.0 + 1e-5
        z = -dist_sq / temperature
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)
