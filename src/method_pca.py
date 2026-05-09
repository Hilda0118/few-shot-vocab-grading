# 把共用的“语义 PCA，保留 95% 方差”抽出来，供所有算法 import
import numpy as np
from sklearn.decomposition import PCA

class SemanticPCA:
    """
    只对前 300 维 FastText 向量做 PCA，并whiten。
    后两维手工特征不参与压缩。
    """
    def __init__(self, seed=0):
        self.pca = PCA(n_components=0.95, whiten=True, random_state=seed)

    # X shape:(n,302)  -> (n,d+2)
    def fit(self, X):
        self.pca.fit(X[:, :-2])

    def transform(self, X):
        Z = self.pca.transform(X[:, :-2])
        return np.hstack([Z, X[:, -2:]])

    def fit_transform(self, X):
        Z = self.pca.fit_transform(X[:, :-2])
        return np.hstack([Z, X[:, -2:]])
