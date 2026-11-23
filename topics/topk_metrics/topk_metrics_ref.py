"""Recall@K 与 NDCG@K 的参考实现。手写时不要看；写完后用 `uv run drill diff topk_metrics --ref` 对照。"""

import math


def recall_at_k(ranked_items, relevant_items, k):
    relevant = set(relevant_items)  # 转成集合，判断是否命中是 O(1)
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    # 分母是相关物品的总数。除以 k 得到的是 Precision@K，不是 Recall@K
    return hits / len(relevant)


def ndcg_at_k(ranked_items, relevant_items, k):
    relevant = set(relevant_items)
    # 位置从 1 数起时折损是 1 / log2(位置 + 1)；下标从 0 数起，所以是 log2(index + 2)。
    # 第一个位置的折损是 1 / log2(2) = 1，即不打折。
    dcg = sum(1.0 / math.log2(index + 2) for index, item in enumerate(ranked_items[:k]) if item in relevant)
    # 理想排序：相关物品全部排在最前面。最多只能放进 k 个，相关物品不足 k 个时就只有这么多。
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg
