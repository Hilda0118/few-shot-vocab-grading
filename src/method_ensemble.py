import numpy as np
from .method_knn import KNNMethod
from .method_logreg import LogRegMethod
from .method_gmm import GMMMethod
from .config import CFG

class EnsembleMethod:
    name = "Ensemble"

    def __init__(self, seed=0):
        self.seed = seed
        self.logreg = LogRegMethod(seed)    # 判别式：LogReg
        self.gmm    = GMMMethod(seed)   # 生成式：GMM
        self.proto  = KNNMethod()       # 度量式：ProtoNet
        self.classes = list(CFG.all_classes)

    def fit_base(self, X_base, y_base):
        self.logreg.fit_base(X_base, y_base)
        self.gmm.fit_base(X_base, y_base)
        self.proto.fit_base(X_base, y_base)

    def predict_base(self, X):
        return self.logreg.predict_base(X)

    def adapt(self, X_support, y_support, X_val, y_val):
        self.logreg.adapt(X_support, y_support, X_val, y_val)
        self.gmm.adapt(X_support, y_support)        # GMM 不需要 val
        self.proto.adapt(X_support, y_support)      # ProtoNet 不需要 val

    def predict(self, X):
        score = self.predict_score(X)
        idx = np.argmax(score, axis=1)
        return np.array([self.classes[i] for i in idx])

    def predict_score(self, X):
        s_logreg = self.logreg.predict_score(X)   # 判别式
        s_gmm    = self.gmm.predict_score(X)       # 生成式
        s_proto  = self.proto.predict_score(X)     # 度量式

        # 【修改】三路融合：LogReg 0.5 + GMM 0.3 + ProtoNet 0.2
        # 三种机制完全不同，组合后才有真正的互补效果
        # LogReg 权重最高因为它用了 val 集调参；GMM 次之；ProtoNet 最低
        avg = 0.5 * s_logreg + 0.3 * s_gmm + 0.2 * s_proto
        avg = avg / (avg.sum(axis=1, keepdims=True) + 1e-12)
        return avg