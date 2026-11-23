"""Recall@K 与 NDCG@K 的测试。"""

import math
import random

import pytest

from topk_metrics import ndcg_at_k, recall_at_k
from topk_metrics_ref import ndcg_at_k as reference_ndcg
from topk_metrics_ref import recall_at_k as reference_recall


def recall_by_definition(ranked_items, relevant_items, k):
    hits = 0
    for position in range(1, k + 1):
        if position <= len(ranked_items) and ranked_items[position - 1] in relevant_items:
            hits += 1
    return hits / len(relevant_items)


def ndcg_by_definition(ranked_items, relevant_items, k):
    """逐个位置（从 1 数起）累加收益；理想排序是把相关物品排在最前面的那个列表。"""
    def dcg(items):
        total = 0.0
        for position in range(1, k + 1):
            if position <= len(items) and items[position - 1] in relevant_items:
                total += 1.0 / math.log2(position + 1)
        return total

    ideal_ranking = list(relevant_items)
    return dcg(ranked_items) / dcg(ideal_ranking)


def random_case(seed):
    rng = random.Random(seed)
    num_items = rng.randint(5, 40)
    ranked = rng.sample(range(100), num_items)
    # 相关物品一部分在推荐列表里，一部分不在
    relevant = set(rng.sample(ranked, rng.randint(0, min(6, num_items)))) | set(rng.sample(range(100, 120), rng.randint(0, 4)))
    if not relevant:
        relevant = {ranked[0]}
    # 一半的用例取很小的 k，保证“相关物品多于 k 个”的情况经常出现
    k = rng.choice([rng.randint(1, 3), rng.randint(1, num_items + 5)])
    return ranked, relevant, k


# ---------- Recall@K ----------

def test_recall_hand_computed():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 50, 99}
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(1 / 3), "前 3 个命中 1 个，相关物品共 3 个，应为 1/3"
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3), "前 5 个命中 2 个，相关物品共 3 个，应为 2/3"
    assert recall_at_k(ranked, relevant, 1) == pytest.approx(0.0), "第 1 个没有命中，应为 0"


def test_recall_denominator_is_number_of_relevant_items():
    # 相关物品只有 1 个且排在第 1 位：Recall@5 = 1，而不是 1/5
    assert recall_at_k([7, 1, 2, 3, 4], {7}, 5) == pytest.approx(1.0), "分母应是相关物品的总数"
    # 相关物品有 4 个，前 2 个全部命中：Recall@2 = 2/4，而不是 1
    assert recall_at_k([1, 2, 3, 4], {1, 2, 3, 4}, 2) == pytest.approx(0.5), "相关物品多于 k 个时，分母仍是相关物品的总数"


def test_recall_only_counts_top_k():
    assert recall_at_k([1, 2, 3, 4, 5], {5}, 4) == pytest.approx(0.0), "排在第 k 个之后的物品不应计入命中"


def test_recall_k_larger_than_list():
    assert recall_at_k([1, 2, 3], {2, 9}, 10) == pytest.approx(0.5), "k 大于推荐列表长度时结果不对"


def test_recall_accepts_list_of_relevant_items():
    assert recall_at_k([1, 2, 3, 4], [4, 2, 8], 3) == pytest.approx(1 / 3), "relevant_items 是列表时结果不对"


@pytest.mark.parametrize("seed", range(8))
def test_recall_matches_definition(seed):
    ranked, relevant, k = random_case(seed)
    assert recall_at_k(ranked, relevant, k) == pytest.approx(recall_by_definition(ranked, relevant, k)), (
        "Recall 与按定义计算的结果不一致"
    )


# ---------- NDCG@K ----------

def test_ndcg_hand_computed():
    # 命中位置 1 和 3：DCG = 1 + 1/log2(4) = 1.5；相关物品 2 个，IDCG = 1 + 1/log2(3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k([5, 6, 7, 8], {5, 7}, 4) == pytest.approx(expected), "NDCG 的手算值不对"


def test_ndcg_single_hit_at_each_position():
    for position in range(1, 6):
        ranked = [0, 1, 2, 3, 4]
        relevant = {ranked[position - 1]}
        assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(1 / math.log2(position + 1)), (
            f"唯一的相关物品排在第 {position} 位时，NDCG 应为 1/log2({position + 1})"
        )


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([3, 1, 2, 9, 8], {1, 2, 3}, 5) == pytest.approx(1.0), "相关物品全部排在最前面时 NDCG 应为 1"
    assert ndcg_at_k([3, 1, 2, 9, 8], {1, 2, 3}, 2) == pytest.approx(1.0), (
        "相关物品多于 k 个且前 k 个全部命中时 NDCG 应为 1"
    )


def test_ndcg_fewer_relevant_items_than_k():
    # 只有 1 个相关物品且排第 1：理想排序也只能命中 1 个，NDCG@5 = 1
    assert ndcg_at_k([4, 1, 2, 3, 5], {4}, 5) == pytest.approx(1.0), "相关物品不足 k 个时，理想排序里也只有这么多个"


def test_ndcg_no_hit_is_zero():
    assert ndcg_at_k([1, 2, 3], {7, 8}, 3) == pytest.approx(0.0), "前 k 个没有命中时 NDCG 应为 0"


def test_ndcg_only_counts_top_k():
    assert ndcg_at_k([1, 2, 3, 4, 5], {5}, 4) == pytest.approx(0.0), "排在第 k 个之后的物品不应计入 DCG"
    # 命中位置 2；位置 4 的命中在 k = 3 之外。相关物品 2 个，IDCG@3 = 1 + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([9, 1, 8, 2], {1, 2}, 3) == pytest.approx(expected), "NDCG@3 的手算值不对"


def test_ndcg_earlier_hit_scores_higher():
    early = ndcg_at_k([1, 9, 8, 7], {1, 5}, 4)
    late = ndcg_at_k([9, 8, 7, 1], {1, 5}, 4)
    assert early > late > 0, "同样命中一个物品，排得越靠前 NDCG 越高"


def test_ndcg_k_larger_than_list():
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    assert ndcg_at_k([8, 2], {1, 2, 3}, 10) == pytest.approx(expected), "k 大于推荐列表长度时结果不对"


@pytest.mark.parametrize("seed", range(8))
def test_ndcg_matches_definition(seed):
    ranked, relevant, k = random_case(seed)
    assert ndcg_at_k(ranked, relevant, k) == pytest.approx(ndcg_by_definition(ranked, relevant, k)), (
        "NDCG 与按定义计算的结果不一致"
    )


@pytest.mark.parametrize("seed", range(20, 30))
def test_matches_reference(seed):
    ranked, relevant, k = random_case(seed)
    assert recall_at_k(ranked, relevant, k) == pytest.approx(reference_recall(ranked, relevant, k)), "Recall 与参考实现不一致"
    assert ndcg_at_k(ranked, list(relevant), k) == pytest.approx(reference_ndcg(ranked, relevant, k)), "NDCG 与参考实现不一致"
