"""K-means 的参考实现。手写时不要看；写完后用 `uv run drill diff kmeans --ref` 对照。"""

import numpy as np


def assign(X, centers):
    # 广播：[n, 1, d] - [1, k, d] -> [n, k, d]，在最后一维上求平方和得到 [n, k]
    # 比较大小不需要开方；argmin 在距离相同时返回最小的编号
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def kmeans(X, init_centers, max_iters=100):
    X = np.asarray(X, dtype=float)
    centers = np.array(init_centers, dtype=float)  # np.array 会复制，不改动调用者传入的数组
    for _ in range(max_iters):
        labels = assign(X, centers)
        new_centers = centers.copy()  # 先复制：空簇的中心保持不变
        for j in range(len(centers)):
            members = X[labels == j]
            if len(members) > 0:  # 空簇直接取均值会得到 nan
                new_centers[j] = members.mean(axis=0)  # 对样本取平均，保留特征维
        if np.array_equal(new_centers, centers):  # 中心不再变化，之后每一轮的结果都相同
            break
        centers = new_centers
    # 循环里的 labels 是按更新前的中心分配的，返回前要按最终的中心再分配一次
    return centers, assign(X, centers)
