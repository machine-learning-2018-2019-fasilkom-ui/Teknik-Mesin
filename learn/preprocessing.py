import numpy as np


class MinMaxScaler:

    def __init__(self):
        self.min_vect = None
        self.max_vect = None

    def fit(self, A):
        A = np.array(A).astype(float)
        self.min_vect = np.min(A, axis=0)
        self.max_vect = np.max(A, axis=0)

    def transform(self, A):
        A = np.array(A).astype(float)
        for j in range(A.shape[1]):
            A[:, j] = (A[:, j] - self.min_vect[j]) / (self.max_vect[j] - self.min_vect[j])
        return A

    def fit_transform(self, A):
        self.fit(A)
        return self.transform(A)
