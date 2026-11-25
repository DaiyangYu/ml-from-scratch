"""AUC 与 GAUC。要求见同目录下的 README.md。"""

import numpy as np


def auc(labels, scores):  # 均为 [n] -> 浮点数
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.argsort(scores,kind = 'stable')
    sorted_scores = scores[order]
    
    bounderies = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])+1
    starts = np.concat([[0],bounderies])
    ends = np.concat([bounderies,[n]])
    average_ranks = (ends+starts+1)/2
    ranks = np.empty(n)
    ranks[order] = np.repeat(average_ranks,ends-starts)

    positive = labels == 1
    n_pos = positive.sum()
    n_neg = n-n_pos
    return float((ranks[positive].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))
    



def gauc(labels, scores, user_ids):  # 均为 [n] -> 浮点数
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    total_sum,total_weight = 0.0,0.0

    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        user_labels = labels[mask]
        n_pos = int((user_labels == 1).sum())
        if n_pos == 0 or n_pos == len(user_labels):
            continue
        weight = len(user_labels)
        total_sum += weight*auc(labels[mask],scores[mask])
        total_weight+=weight
        
    return float(total_sum/total_weight)
