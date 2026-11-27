"""AUC 与 GAUC。要求见同目录下的 README.md。"""

import numpy as np


def auc(labels, scores):  # 均为 [n] -> 浮点数
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])+1
    starts = np.concat([[0], boundaries])
    ends = np.concat([boundaries,[n]])
    average_ranks = (starts+ends+1)/2
    ranks = np.empty(n)
    ranks[order] = np.repeat(average_ranks,ends-starts)

    positive = labels == 1
    n_pos = positive.sum()
    n_neg = n-n_pos
    return float((ranks[positive].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def gauc(labels, scores, user_ids):  # 均为 [n] -> 浮点数
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    weighted_sum = 0.0
    total_weight = 0

    for user in np.unique(user_ids):
        mask = user_ids == user
        user_labels = labels[mask]
        n_pos = (user_labels == 1).sum()
        weight = len(user_labels)
        if n_pos == weight or n_pos == 0:
            continue
        weighted_sum+=(weight*auc(user_labels,scores[mask]))
        total_weight+=weight
    return float(weighted_sum/total_weight)
