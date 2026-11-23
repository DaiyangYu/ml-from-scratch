"""批内负采样 Softmax 损失的测试。"""

import math

import pytest
import torch

from drill.testing import assert_close
from inbatch_softmax import inbatch_softmax_loss
from inbatch_softmax_ref import inbatch_softmax_loss as reference_loss


def by_definition(user_vecs, item_vecs, temperature, item_log_probs=None):
    """逐个用户计算：-log( exp(s_ii) / Σ_j exp(s_ij) )，其中 s_ij = <u_i, v_j> / 温度 − log p_j。"""
    batch = user_vecs.size(0)
    total = 0.0
    for i in range(batch):
        scores = []
        for j in range(batch):
            score = torch.dot(user_vecs[i], item_vecs[j]).item() / temperature
            if item_log_probs is not None:
                score -= item_log_probs[j].item()
            scores.append(score)
        highest = max(scores)
        log_sum = highest + math.log(sum(math.exp(s - highest) for s in scores))
        total += log_sum - scores[i]
    return torch.tensor(total / batch)


def random_case(seed, batch=6, dim=5):
    generator = torch.Generator().manual_seed(seed)
    user_vecs = torch.randn(batch, dim, generator=generator)
    item_vecs = torch.randn(batch, dim, generator=generator)
    # 采样概率：先取随机正数再归一化，各物品相差较大
    probs = torch.rand(batch, generator=generator) ** 3 + 1e-3
    log_probs = torch.log(probs / probs.sum())
    return user_vecs, item_vecs, log_probs


def test_loss_is_scalar():
    user_vecs, item_vecs, _ = random_case(0)
    loss = inbatch_softmax_loss(user_vecs, item_vecs)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "损失应为 0 维的标量张量"


def test_hand_computed_value():
    # 用户与物品向量都是单位矩阵，温度为 1：每行的打分是 [1, 0] 或 [0, 1]，损失 = log(e + 1) − 1
    eye = torch.eye(2)
    loss = inbatch_softmax_loss(eye, eye, temperature=1.0)
    assert_close(loss, torch.tensor(math.log(math.e + 1) - 1), "损失的手算值")


def test_temperature_divides_scores():
    # 同样的向量，温度 0.5：打分是 [2, 0]，损失 = log(e² + 1) − 2
    eye = torch.eye(2)
    loss = inbatch_softmax_loss(eye, eye, temperature=0.5)
    assert_close(loss, torch.tensor(math.log(math.e ** 2 + 1) - 2), "温度为 0.5 时的损失")


def test_default_temperature():
    user_vecs, item_vecs, _ = random_case(1)
    user_vecs, item_vecs = user_vecs * 0.2, item_vecs * 0.2
    assert_close(inbatch_softmax_loss(user_vecs, item_vecs), by_definition(user_vecs, item_vecs, 0.05), "默认温度（0.05）下的损失", atol=1e-4)


def test_vectors_are_not_normalized_inside():
    # 把向量整体放大 3 倍，内积放大 9 倍；如果函数内部做了归一化，损失不会变
    user_vecs, item_vecs, _ = random_case(2)
    scaled = inbatch_softmax_loss(3 * user_vecs, 3 * item_vecs, temperature=1.0)
    assert_close(scaled, by_definition(3 * user_vecs, 3 * item_vecs, 1.0), "向量放大后的损失（函数内部不应做归一化）")


@pytest.mark.parametrize("seed, batch, dim, temperature", [(3, 6, 5, 1.0), (4, 2, 3, 0.5), (5, 9, 4, 0.2), (6, 1, 4, 1.0)])
def test_matches_definition(seed, batch, dim, temperature):
    user_vecs, item_vecs, _ = random_case(seed, batch, dim)
    assert_close(
        inbatch_softmax_loss(user_vecs, item_vecs, temperature=temperature),
        by_definition(user_vecs, item_vecs, temperature),
        "损失",
        atol=1e-4,
    )


def test_rows_are_users():
    # 用户 0 分不清两个物品，用户 1 分得很清：按行（每个用户在物品中选）与按列（每个物品在用户中选）结果不同
    user_vecs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    item_vecs = torch.tensor([[1.0, 0.0], [1.0, 5.0]])
    loss = inbatch_softmax_loss(user_vecs, item_vecs, temperature=1.0)
    assert_close(loss, by_definition(user_vecs, item_vecs, 1.0), "损失（每个用户应在 batch 内的全部物品中做选择）")


@pytest.mark.parametrize("seed, batch, dim, temperature", [(7, 6, 5, 1.0), (8, 3, 4, 0.5), (9, 8, 6, 0.1)])
def test_correction_matches_definition(seed, batch, dim, temperature):
    user_vecs, item_vecs, log_probs = random_case(seed, batch, dim)
    assert_close(
        inbatch_softmax_loss(user_vecs, item_vecs, temperature=temperature, item_log_probs=log_probs),
        by_definition(user_vecs, item_vecs, temperature, log_probs),
        "带纠偏的损失",
        atol=1e-4,
    )


def test_correction_changes_the_loss():
    user_vecs, item_vecs, log_probs = random_case(10)
    plain = inbatch_softmax_loss(user_vecs, item_vecs, temperature=1.0)
    corrected = inbatch_softmax_loss(user_vecs, item_vecs, temperature=1.0, item_log_probs=log_probs)
    assert abs(plain.item() - corrected.item()) > 1e-3, "各物品采样概率不同时，纠偏应当改变损失"


def test_uniform_probabilities_change_nothing():
    user_vecs, item_vecs, _ = random_case(11)
    plain = inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.5)
    uniform = torch.full((user_vecs.size(0),), math.log(0.01))
    corrected = inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.5, item_log_probs=uniform)
    assert_close(corrected, plain, "所有物品采样概率相同时的损失（应与不纠偏时相同）")


def test_popular_negative_is_penalized_less():
    # 用户 0 对物品 0（正样本）和物品 1（负样本）的内积相同，不纠偏时它的损失是 ln 2。
    # 用户 1 对自己的正样本有极大的优势，损失恒为 0，不受纠偏影响。所以整批损失 = 用户 0 的损失 / 2。
    user_vecs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    item_vecs = torch.tensor([[1.0, 0.0], [1.0, 1000.0]])
    plain = inbatch_softmax_loss(user_vecs, item_vecs, temperature=1.0).item()
    popular_negative = inbatch_softmax_loss(
        user_vecs, item_vecs, temperature=1.0, item_log_probs=torch.log(torch.tensor([0.1, 0.8]))).item()
    rare_negative = inbatch_softmax_loss(
        user_vecs, item_vecs, temperature=1.0, item_log_probs=torch.log(torch.tensor([0.1, 0.01]))).item()
    assert plain == pytest.approx(math.log(2) / 2, abs=1e-5), "不纠偏时的损失应为 ln 2 / 2"
    assert popular_negative < plain < rare_negative, (
        "负样本越热门，纠偏后它带来的损失应当越小："
        f"热门负样本 {popular_negative:.4f}，不纠偏 {plain:.4f}，冷门负样本 {rare_negative:.4f}"
    )
    # 正样本的采样概率 0.1，负样本 0.8：正样本的概率 = (1/0.1) / (1/0.1 + 1/0.8)
    assert popular_negative == pytest.approx(-math.log(10 / 11.25) / 2, abs=1e-5), "热门负样本时的损失手算值不对"


def test_gradient_reaches_both_towers():
    user_vecs, item_vecs, log_probs = random_case(12)
    user_vecs.requires_grad_(True)
    item_vecs.requires_grad_(True)
    inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.5, item_log_probs=log_probs).backward()
    assert user_vecs.grad is not None and user_vecs.grad.abs().sum() > 0, "梯度没有传回用户向量"
    assert item_vecs.grad is not None and item_vecs.grad.abs().sum() > 0, "梯度没有传回物品向量"


@pytest.mark.parametrize("seed, with_correction", [(20, False), (21, True), (22, False), (23, True)])
def test_matches_reference(seed, with_correction):
    user_vecs, item_vecs, log_probs = random_case(seed, batch=16, dim=8)
    log_probs = log_probs if with_correction else None
    assert_close(
        inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.3, item_log_probs=log_probs),
        reference_loss(user_vecs, item_vecs, temperature=0.3, item_log_probs=log_probs),
        "损失",
        atol=1e-4,
    )
