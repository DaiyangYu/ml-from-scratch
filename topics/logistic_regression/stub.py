"""逻辑回归。要求见同目录下的 README.md。"""

import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        raise NotImplementedError

    def fit(self, X, y):  # X: [n, d], y: [n] -> self
        raise NotImplementedError

    def predict_proba(self, X):  # [n, d] -> [n]
        raise NotImplementedError

    def predict(self, X):  # [n, d] -> [n]，取值 0 或 1
        raise NotImplementedError
