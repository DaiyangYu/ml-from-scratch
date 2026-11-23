"""AUC 与 GAUC 的参考实现。手写时不要看；写完后用 `uv run drill diff auc --ref` 对照。"""

import numpy as np


def auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    # 把全部样本按分数从低到高排名，名次从 1 开始。排序是整个算法里唯一的 O(n log n) 步骤。
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    # 分数相同的样本共用它们所占名次的平均值，这样一对同分的正负样本恰好贡献 0.5，
    # 结果也不依赖同分样本在输入中的先后顺序。
    boundaries = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1  # 每一组同分样本的起点
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    average_rank = (starts + 1 + ends) / 2  # 这一组占据第 starts+1 到第 ends 名，取平均
    ranks = np.empty(n)
    ranks[order] = np.repeat(average_rank, ends - starts)

    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = n - n_pos
    # 一个正样本的名次 = 分数低于它的样本数 + 1。把所有正样本的名次加起来，
    # 其中正样本彼此之间的比较占了 n_pos(n_pos+1)/2，减掉之后剩下的就是
    # “正样本排在负样本之上”的对数；再除以正负样本对的总数 n_pos * n_neg。
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gauc(labels, scores, user_ids):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    user_ids = np.asarray(user_ids)
    weighted_sum, total_weight = 0.0, 0.0
    for user in np.unique(user_ids):
        mask = user_ids == user
        user_labels = labels[mask]
        n_pos = int((user_labels == 1).sum())
        if n_pos == 0 or n_pos == len(user_labels):
            continue  # 只有一类样本的用户没有正负样本对，AUC 无定义：跳过，也不计入权重
        weight = len(user_labels)  # 按该用户的样本数（曝光数）加权，不是各用户简单平均
        weighted_sum += weight * auc(user_labels, scores[mask])
        total_weight += weight
    return float(weighted_sum / total_weight)
