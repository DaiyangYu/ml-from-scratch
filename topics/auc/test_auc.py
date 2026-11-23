"""AUC 与 GAUC 的测试。"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auc import auc, gauc
from auc_ref import auc as reference_auc
from auc_ref import gauc as reference_gauc


def auc_by_definition(labels, scores):
    """按定义统计：所有（正样本，负样本）对中，正样本分数更高的比例，同分算 0.5。"""
    wins, pairs = 0.0, 0
    for label_a, score_a in zip(labels, scores):
        if label_a != 1:
            continue
        for label_b, score_b in zip(labels, scores):
            if label_b == 1:
                continue
            pairs += 1
            if score_a > score_b:
                wins += 1
            elif score_a == score_b:
                wins += 0.5
    return wins / pairs


def gauc_by_definition(labels, scores, user_ids):
    weighted_sum, total_weight = 0.0, 0
    for user in sorted(set(user_ids)):
        rows = [i for i, u in enumerate(user_ids) if u == user]
        user_labels = [labels[i] for i in rows]
        if len(set(user_labels)) < 2:
            continue
        weighted_sum += len(rows) * auc_by_definition(user_labels, [scores[i] for i in rows])
        total_weight += len(rows)
    return weighted_sum / total_weight


def random_case(seed, n=60, levels=None, positive_rate=0.4):
    """levels 不为 None 时把分数限制在少数几个取值上，制造大量同分样本。"""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < positive_rate).astype(int)
    labels[0], labels[1] = 1, 0  # 保证两类都存在
    scores = rng.random(n)
    if levels is not None:
        scores = np.floor(scores * levels) / levels
    return labels, scores


# ---------- AUC ----------

def test_auc_returns_float():
    result = auc(np.array([1, 0]), np.array([0.9, 0.1]))
    assert isinstance(result, float), f"应返回浮点数，实际为 {type(result).__name__}"


def test_auc_hand_computed():
    # 正样本 0.9、0.3，负样本 0.8、0.1：四对中 (0.9,0.8) (0.9,0.1) (0.3,0.1) 三对正样本更高
    assert auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.3, 0.1])) == pytest.approx(0.75), "AUC 应为 3/4"


def test_auc_perfect_and_reversed():
    labels = np.array([0, 0, 1, 1, 1])
    assert auc(labels, np.array([0.1, 0.2, 0.7, 0.8, 0.9])) == pytest.approx(1.0), "正样本全部排在负样本之上时 AUC 应为 1"
    assert auc(labels, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(0.0), "正样本全部排在负样本之下时 AUC 应为 0"


def test_auc_all_scores_equal():
    assert auc(np.array([1, 0, 0, 1, 0]), np.full(5, 0.3)) == pytest.approx(0.5), "所有分数相同时 AUC 应为 0.5"


def test_auc_tied_pair_counts_half():
    # 正样本 0.5；负样本 0.5、0.2：一对同分（0.5），一对正样本更高（1）
    assert auc(np.array([1, 0, 0]), np.array([0.5, 0.5, 0.2])) == pytest.approx(0.75), "同分的一对应按 0.5 计"
    assert auc(np.array([0, 1, 0]), np.array([0.5, 0.5, 0.2])) == pytest.approx(0.75), "结果不应依赖同分样本的先后顺序"


def test_auc_accepts_lists():
    assert auc([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1]) == pytest.approx(0.75), "输入是 Python 列表时结果不对"


def test_auc_integer_scores():
    assert auc(np.array([1, 0, 1, 0]), np.array([3, 2, 2, 1])) == pytest.approx(0.875), "分数是整数时结果不对"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_auc_matches_definition(seed):
    labels, scores = random_case(seed)
    assert auc(labels, scores) == pytest.approx(auc_by_definition(labels, scores), abs=1e-12), "与按定义统计的结果不一致"


@pytest.mark.parametrize("seed, levels", [(4, 2), (5, 3), (6, 5), (7, 10)])
def test_auc_matches_definition_with_many_ties(seed, levels):
    labels, scores = random_case(seed, n=80, levels=levels)
    assert auc(labels, scores) == pytest.approx(auc_by_definition(labels, scores), abs=1e-12), (
        "存在大量同分样本时，与按定义统计的结果不一致"
    )


@pytest.mark.parametrize("positive_rate", [0.03, 0.97])
def test_auc_unbalanced(positive_rate):
    labels, scores = random_case(8, n=150, levels=20, positive_rate=positive_rate)
    assert auc(labels, scores) == pytest.approx(auc_by_definition(labels, scores), abs=1e-12), (
        "正负样本数量相差很大时结果不对"
    )


def test_auc_does_not_depend_on_sample_order():
    labels, scores = random_case(9, n=100, levels=4)
    permutation = np.random.default_rng(0).permutation(len(labels))
    assert auc(labels[permutation], scores[permutation]) == pytest.approx(auc(labels, scores), abs=1e-12), (
        "打乱样本顺序后结果变了"
    )


LARGE_CASE = """
import resource
import numpy as np
from auc import auc
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
rng = np.random.default_rng(0)
labels = rng.integers(0, 2, 200_000)
scores = np.round(rng.random(200_000), 4)
print(repr(float(auc(labels, scores))))
"""


def test_auc_large_input_is_fast_enough():
    """20 万个样本：两两比较有 10^10 对，循环写法算不完，广播写法需要约 10 GB 内存。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", LARGE_CASE],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("20 万个样本在 10 秒内没有算完，需要 O(n log n) 的写法")
    error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
    assert result.returncode == 0, f"20 万个样本时运行失败（内存上限 4 GB），需要 O(n log n) 的写法：{error}"
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200_000)
    scores = np.round(rng.random(200_000), 4)
    value = float(result.stdout.strip().splitlines()[-1])
    assert value == pytest.approx(reference_auc(labels, scores), abs=1e-9), "20 万个样本时结果不对"


@pytest.mark.parametrize("seed, levels", [(10, None), (11, 7), (12, 50)])
def test_auc_matches_reference(seed, levels):
    labels, scores = random_case(seed, n=5000, levels=levels)
    assert auc(labels, scores) == pytest.approx(reference_auc(labels, scores), abs=1e-12), "与参考实现不一致"


# ---------- GAUC ----------

def test_gauc_returns_float():
    result = gauc(np.array([1, 0]), np.array([0.9, 0.1]), np.array([7, 7]))
    assert isinstance(result, float), f"应返回浮点数，实际为 {type(result).__name__}"


def test_gauc_hand_computed_weighting():
    # 用户 1 有 2 个样本，AUC = 1；用户 2 有 4 个样本，AUC = 0.25。
    # 按样本数加权：(2 * 1 + 4 * 0.25) / 6 = 0.5；简单平均会得到 0.625。
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.4])
    user_ids = np.array([1, 1, 2, 2, 2, 2])
    assert gauc(labels, scores, user_ids) == pytest.approx(0.5), "应按各用户的样本数加权平均"


def test_gauc_skips_single_class_users():
    # 用户 1：AUC = 1（2 个样本）；用户 2 只有正样本；用户 3 只有负样本；用户 4：AUC = 0（2 个样本）
    labels = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.5, 0.6, 0.7, 0.3, 0.2, 0.8, 0.4])
    user_ids = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4])
    assert gauc(labels, scores, user_ids) == pytest.approx(0.5), (
        "只有一类样本的用户应当跳过，既不计入加权和，也不计入权重之和"
    )


def test_gauc_differs_from_global_auc():
    # 每个用户内部排序完全正确，但不同用户的分数水平不同：GAUC 为 1，全局 AUC 小于 1
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.2, 0.1, 0.9, 0.8])
    user_ids = np.array([1, 1, 2, 2])
    assert gauc(labels, scores, user_ids) == pytest.approx(1.0), "每个用户内部排序都正确时 GAUC 应为 1"


def test_gauc_users_interleaved_and_lists():
    labels = [1, 0, 0, 1, 0, 1, 1, 0]
    scores = [0.3, 0.9, 0.2, 0.3, 0.6, 0.1, 0.8, 0.5]
    user_ids = [5, 9, 5, 9, 5, 9, 5, 9]
    assert gauc(labels, scores, user_ids) == pytest.approx(gauc_by_definition(labels, scores, user_ids)), (
        "同一用户的样本不相邻、输入是 Python 列表时结果不对"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gauc_matches_definition(seed):
    rng = np.random.default_rng(seed)
    n = 120
    user_ids = rng.integers(0, 15, n)
    labels = (rng.random(n) < 0.3).astype(int)
    scores = np.floor(rng.random(n) * 6) / 6
    expected = gauc_by_definition(labels.tolist(), scores.tolist(), user_ids.tolist())
    assert gauc(labels, scores, user_ids) == pytest.approx(expected, abs=1e-12), "与按定义计算的结果不一致"


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_gauc_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    user_ids = rng.integers(0, 200, n)
    labels = (rng.random(n) < 0.2).astype(int)
    scores = np.round(rng.random(n), 2)
    assert gauc(labels, scores, user_ids) == pytest.approx(reference_gauc(labels, scores, user_ids), abs=1e-12), (
        "与参考实现不一致"
    )
