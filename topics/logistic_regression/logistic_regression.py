"""逻辑回归。要求见同目录下的 README.md。"""

import numpy as np

def sigmoid(z):
    return np.exp(-np.logaddexp(0.0,-z))


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):  # X: [n, d], y: [n] -> self
        X = np.asarray(X)
        y = np.asarray(y)
        n,d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            p = sigmoid(X@self.w+self.b)
            error = p-y
            grad_w = 1/n * X.T @ error
            grad_b = error.mean()
            self.w -= self.lr*grad_w
            self.b -= self.lr*grad_b
        return self

    def predict_proba(self, X):  # [n, d] -> [n]
        return sigmoid(np.asarray(X,dtype=float)@self.w + self.b)

    def predict(self, X):  # [n, d] -> [n]，取值 0 或 1
        return (self.predict_proba(X) >= 0.5).astype(int)
