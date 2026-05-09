import numpy as np
from .config import CFG


class FixedAnchorMethod:
    name = "FixedAnchor"

    def __init__(self, alpha=5.0):
        self.classes = list(CFG.all_classes)
        self.anchors = {}
        self.alpha = alpha
        self.base_mean = None

    def _process_features(self, X):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base)
        self.base_mean = self.X_base_proc.mean(axis=0)
        self.base_mean /= (np.linalg.norm(self.base_mean) + 1e-10)
        for c in CFG.base_classes:
            p = self.X_base_proc[y_base == c].mean(axis=0)
            self.anchors[c] = p / (np.linalg.norm(p) + 1e-10)
        for c in CFG.novel_classes:
            self.anchors[c] = self.base_mean.copy()

    def adapt(self, X_support, y_support):
        X_support_proc = self._process_features(X_support)
        for c in CFG.novel_classes:
            xc = X_support_proc[y_support == c]
            if len(xc) == 0:
                self.anchors[c] = self.base_mean.copy()
                continue
            a_novel = xc.mean(axis=0)
            a_novel /= (np.linalg.norm(a_novel) + 1e-10)
            k = len(xc)
            lam = k / (k + self.alpha)
            p = lam * a_novel + (1.0 - lam) * self.base_mean
            self.anchors[c] = p / (np.linalg.norm(p) + 1e-10)

    def predict(self, X):
        X_proc = self._process_features(X)
        # 【修改】L2归一化后用余弦相似度（内积），而非欧式距离
        # 原因：特征已经 L2 归一化，欧式距离 = sqrt(2-2*cos)，单调等价
        # 但直接用内积数值更稳定，且 shot 增加时 anchor 更新方向更准确
        Xn = X_proc / (np.linalg.norm(X_proc, axis=1, keepdims=True) + 1e-10)
        A  = np.vstack([self.anchors[c] for c in self.classes])
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
        sim = Xn @ An.T   # 余弦相似度
        idx = np.argmax(sim, axis=1)
        return np.array([self.classes[i] for i in idx])

    def predict_score(self, X):
        X_proc = self._process_features(X)
        Xn = X_proc / (np.linalg.norm(X_proc, axis=1, keepdims=True) + 1e-10)
        A  = np.vstack([self.anchors[c] for c in self.classes])
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
        sim = Xn @ An.T
        # 【修改】固定温度 softmax，避免动态温度在高维下数值不稳定
        z = sim / 0.1   # 温度=0.1，锐化分布，让高相似度类别更突出
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)