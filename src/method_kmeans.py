import numpy as np
from sklearn.cluster import KMeans
from .config import CFG

class KMeansMethod:
    """
    与 FixedAnchor 的本质区别：
      FixedAnchor: novel 中心 = 插值(support均值, base均值)，纯线性插值
      KMeans:      novel 中心 = KMeans 在 support 数据上聚类的自然中心
                   + 用 base 类均值做 warm-start 初始化，防止随机初始化失败

    关键设计：
      1. base 类中心固定为有监督均值（稳定）
      2. novel 类在 support 样本上独立聚类（不混入 base 数据，避免被淹没）
      3. 用 base 全局均值做 warm-start，让聚类从合理位置开始
      4. shot 越多，聚类中心越准确 → 性能单调增长
    """
    name = "KMeans"

    def __init__(self, seed=0):
        self.seed = seed
        self.classes = list(CFG.all_classes)
        self.centers = {}

    def _process_features(self, X):
        return X

    def fit_base(self, X_base, y_base):
        self.X_base_proc = self._process_features(X_base)
        self.y_base = y_base
        self.base_centers = {}
        for c in CFG.base_classes:
            mask = (y_base == c)
            p = self.X_base_proc[mask].mean(axis=0) if mask.sum() > 0 \
                else self.X_base_proc.mean(axis=0)
            self.base_centers[c] = p / (np.linalg.norm(p) + 1e-10)
        # 全局均值，用于 novel 类 warm-start
        self.global_mean = self.X_base_proc.mean(axis=0)
        self.global_mean /= (np.linalg.norm(self.global_mean) + 1e-10)

    def predict_base(self, X):
        X_proc = self._process_features(X)
        Xn = X_proc / (np.linalg.norm(X_proc, axis=1, keepdims=True) + 1e-10)
        protos = np.vstack([self.base_centers[c] for c in CFG.base_classes])
        sim = Xn @ protos.T
        return np.array([CFG.base_classes[i] for i in np.argmax(sim, axis=1)])

    def adapt(self, X_support, y_support):
        X_support_proc = self._process_features(X_support)
        self.centers = {}

        # base 类：有监督均值，固定不变
        for c in CFG.base_classes:
            self.centers[c] = self.base_centers[c]

        # novel 类：在 support 样本上做 KMeans
        n_novel = len(CFG.novel_classes)
        novel_support_mask = np.isin(y_support, CFG.novel_classes)
        X_novel_supp = X_support_proc[novel_support_mask]
        y_novel_supp = y_support[novel_support_mask]

        if len(X_novel_supp) >= n_novel:
            # 有足够样本：用 KMeans 聚类
            # warm-start：用每个 novel 类的 support 均值初始化（如果有的话）
            init_centers = []
            for c in CFG.novel_classes:
                mask = (y_novel_supp == c)
                if mask.sum() > 0:
                    p = X_novel_supp[mask].mean(axis=0)
                else:
                    p = self.global_mean.copy() * (
                        np.linalg.norm(X_novel_supp.mean(axis=0)) + 1e-10
                    )
                init_centers.append(p)
            init_centers = np.vstack(init_centers)

            km = KMeans(
                n_clusters=n_novel,
                init=init_centers,   # warm-start，避免随机初始化
                n_init=1,
                random_state=self.seed
            )
            km.fit(X_novel_supp)

            # 投票：把聚类中心映射到 novel 标签
            labels_pred = km.predict(X_novel_supp)
            cluster_votes = {}
            for cid in range(n_novel):
                mask = (labels_pred == cid)
                if mask.sum() == 0:
                    cluster_votes[cid] = CFG.novel_classes[cid]
                    continue
                vote = {}
                for lbl in y_novel_supp[mask]:
                    vote[lbl] = vote.get(lbl, 0) + 1
                cluster_votes[cid] = max(vote, key=vote.get)

            # 确保每个 novel 类都有中心
            assigned = {v: k for k, v in cluster_votes.items()}
            for i, c in enumerate(CFG.novel_classes):
                if c in assigned:
                    cid = assigned[c]
                    p = km.cluster_centers_[cid]
                else:
                    # 兜底：用 support 均值
                    mask = (y_novel_supp == c)
                    p = X_novel_supp[mask].mean(axis=0) if mask.sum() > 0 \
                        else self.global_mean
                self.centers[c] = p / (np.linalg.norm(p) + 1e-10)

        else:
            # 样本不足：直接用 support 均值（退化为 ProtoNet）
            for c in CFG.novel_classes:
                mask = (y_support == c)
                if mask.sum() > 0:
                    p = X_support_proc[mask].mean(axis=0)
                else:
                    p = self.global_mean.copy()
                self.centers[c] = p / (np.linalg.norm(p) + 1e-10)

    def predict(self, X):
        X_proc = self._process_features(X)
        Xn = X_proc / (np.linalg.norm(X_proc, axis=1, keepdims=True) + 1e-10)
        protos = np.vstack([self.centers[c] for c in self.classes])
        sim = Xn @ protos.T
        return np.array([self.classes[i] for i in np.argmax(sim, axis=1)])

    def predict_score(self, X):
        X_proc = self._process_features(X)
        Xn = X_proc / (np.linalg.norm(X_proc, axis=1, keepdims=True) + 1e-10)
        protos = np.vstack([self.centers[c] for c in self.classes])
        sim = Xn @ protos.T
        # 温度 0.3：比 FixedAnchor(0.1) 更软，保留不确定性，与其区分
        z = sim / 0.3
        z -= np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)