import numpy as np


class PCA:

    def __init__(self, n_components):
        self.mean = None
        self.values = None
        self.vectors = None
        self.n_components = n_components

    def fit(self, A):
        A = np.array(A)
        self.mean = A.mean(0)

        C = A - self.mean
        V = np.cov(C.T)
        _, s, vh = np.linalg.svd(V)

        self.values = s[:self.n_components]
        self.vectors = vh[:self.n_components]

    def transform(self, A):
        A = np.array(A)
        C = A - self.mean
        P = self.vectors.dot(C.T)
        return P.T

    def fit_transform(self, A):
        self.fit(A)
        return self.transform(A)
