"""AUC 与 GAUC。要求见同目录下的 README.md。"""

import numpy as np


def auc(labels, scores):  # 均为 [n] -> 浮点数
    labels = np.asarray(labels)
    scores = np.asarray(scores,dtype=float)
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    n = len(scores)

    boundaries = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])+1
    starts = np.concat([[0],boundaries])
    ends = np.concat([boundaries,[n]])
    average_ranks = (ends+starts+1)/2
    ranks = np.empty(n)
    ranks[sorted_idx] = np.repeat(average_ranks,ends-starts)

    positive = labels == 1
    n_pos = positive.sum()
    n_neg = n-n_pos
    return float((ranks[positive].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))

def gauc(labels, scores, user_ids):  # 均为 [n] -> 浮点数
    scores = np.asarray(scores,dtype=float)
    labels = np.asarray(labels)
    user_ids = np.asarray(user_ids)

    weighted_sum = 0.0
    total_weight = 0.0
    for user in np.unique(user_ids):
        user_idx = user_ids == user
        user_labels = labels[user_idx]
        n = len(user_labels)
        n_pos = user_labels.sum()
        if n_pos == 0 or n_pos == n:
            continue
        user_auc = auc(user_labels,scores[user_idx])
        weighted_sum+=user_auc*n
        total_weight+=n
    weighted_sum/=total_weight
    return float(weighted_sum)

