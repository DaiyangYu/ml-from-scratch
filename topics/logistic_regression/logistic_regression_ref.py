"""逻辑回归的参考实现。手写时不要看；写完后用 `uv run drill diff logistic_regression --ref` 对照。"""

import numpy as np


def sigmoid(z):
    # 1 / (1 + exp(-z)) 的稳定写法：logaddexp(0, -z) = log(1 + exp(-z))，z 是很大的负数时不会溢出。
    # 另一种常见写法 exp(z) / (1 + exp(z)) 在 z 很大时是 inf / inf，得到 nan。
    return np.exp(-np.logaddexp(0.0, -z))


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)  # 每次 fit 都从 0 开始
        self.b = 0.0
        for _ in range(self.n_iters):
            logits = X @ self.w + self.b      # [n]
            p = sigmoid(logits)               # [n] 预测为正类的概率
            # 交叉熵接在 sigmoid 后面，对 logit 的导数化简后就是“预测值 − 标签”
            error = p - y                     # [n]
            # 损失是对样本取平均，所以梯度也要除以 n
            grad_w = X.T @ error / n          # [d]
            grad_b = error.mean()
            self.w -= self.lr * grad_w        # 沿梯度的反方向走
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
