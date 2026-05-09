import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .config import CFG

class LogRegMethod:
    name = "LogReg"

    def __init__(self, seed=0):
        self.seed = seed
        self.model = None

    def _process_features(self, X, is_fit=False):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base, is_fit=True)
        self.y_base = y_base
        self.base_model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=self.seed
        )
        self.base_model.fit(self.X_base_proc, self.y_base)

    def predict_base(self, X):
        return self.base_model.predict(self._process_features(X))

    def adapt(self, X_support, y_support, X_val, y_val):
        X_support_proc = self._process_features(X_support)
        X_val_proc = self._process_features(X_val)

        X = np.vstack([self.X_base_proc, X_support_proc])
        y = np.concatenate([self.y_base, y_support])

        best_f1 = -1
        best_model = None

        for c in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
            clf = LogisticRegression(
                C=c, class_weight="balanced", max_iter=2000, random_state=self.seed
            )
            clf.fit(X, y)
            pred = clf.predict(X_val_proc)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf

        self.model = best_model

    def predict(self, X):
        return self.model.predict(self._process_features(X))

    def predict_score(self, X):
        proba = self.model.predict_proba(self._process_features(X))
        model_classes = list(self.model.classes_)
        out = np.zeros((X.shape[0], 4), dtype=float)
        idx = {c: i for i, c in enumerate(CFG.all_classes)}
        for j, c in enumerate(model_classes):
            out[:, idx[c]] = proba[:, j]
        out = out / (out.sum(axis=1, keepdims=True) + 1e-12)
        return out
