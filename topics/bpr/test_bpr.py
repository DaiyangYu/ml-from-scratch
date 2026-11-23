"""BPR 损失的测试。用一张固定的打分表充当模型，不依赖其他题目。"""

import math

import pytest
import torch

from drill.testing import assert_close
from bpr import bpr_loss
from bpr_ref import bpr_loss as reference_loss


class TableScorer:
    """打分模型的替身：score(u, i) 直接查表 table[u, i]。"""

    def __init__(self, table):
        self.table = table

    def __call__(self, users, items):
        return self.table[users, items]


def by_definition(table, users, pos_items, neg_items):
    """按定义逐样本计算 -log σ(正分 − 负分)，再取平均。"""
    total = 0.0
    for u, p, n in zip(users.tolist(), pos_items.tolist(), neg_items.tolist()):
        difference = (table[u, p] - table[u, n]).item()
        total += math.log1p(math.exp(-difference))
    return torch.tensor(total / users.size(0))


def random_case(seed, num_users=6, num_items=9, batch=8):
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(num_users, num_items, generator=generator) * 2
    users = torch.randint(0, num_users, (batch,), generator=generator)
    pos_items = torch.randint(0, num_items, (batch,), generator=generator)
    neg_items = torch.randint(0, num_items, (batch,), generator=generator)
    return table, users, pos_items, neg_items


def test_loss_is_scalar():
    table, users, pos_items, neg_items = random_case(0)
    loss = bpr_loss(TableScorer(table), users, pos_items, neg_items)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "损失应为 0 维的标量张量"


def test_equal_scores_give_ln2():
    table = torch.full((3, 4), 1.7)
    loss = bpr_loss(TableScorer(table), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.tensor([3, 2, 1]))
    assert_close(loss, torch.tensor(math.log(2.0)), "正负样本分数相等时的损失（应为 ln 2）")


def test_hand_computed_values():
    # 用户 0：正分 3，负分 1，差 2；用户 1：正分 -1，负分 0.5，差 -1.5
    table = torch.tensor([[3.0, 1.0, 0.0], [0.5, 0.0, -1.0]])
    loss = bpr_loss(TableScorer(table), torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([1, 0]))
    expected = (math.log1p(math.exp(-2.0)) + math.log1p(math.exp(1.5))) / 2
    assert_close(loss, torch.tensor(expected), "损失")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_definition(seed):
    table, users, pos_items, neg_items = random_case(seed)
    loss = bpr_loss(TableScorer(table), users, pos_items, neg_items)
    assert_close(loss, by_definition(table, users, pos_items, neg_items), "损失")


def test_single_sample():
    table, users, pos_items, neg_items = random_case(5, batch=1)
    loss = bpr_loss(TableScorer(table), users, pos_items, neg_items)
    assert_close(loss, by_definition(table, users, pos_items, neg_items), "损失")


def test_higher_positive_score_gives_lower_loss():
    users, pos_items, neg_items = torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([1, 1])
    positive_ahead = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    negative_ahead = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    good = bpr_loss(TableScorer(positive_ahead), users, pos_items, neg_items).item()
    bad = bpr_loss(TableScorer(negative_ahead), users, pos_items, neg_items).item()
    assert good < bad, f"正样本分数更高时损失应当更小，实际为 {good:.4f}（正样本领先）与 {bad:.4f}（负样本领先）"
    assert good > 0, "损失应为正数"


def test_gradient_direction():
    table = torch.zeros(2, 3, requires_grad=True)
    loss = bpr_loss(TableScorer(table), torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
    loss.backward()
    assert table.grad is not None, "梯度没有传回打分模型"
    assert table.grad[0, 0] < 0 and table.grad[1, 1] < 0, "提高正样本分数应使损失下降（对正样本分数的梯度应为负）"
    assert table.grad[0, 2] > 0 and table.grad[1, 2] > 0, "提高负样本分数应使损失上升（对负样本分数的梯度应为正）"


def test_large_score_gap_stays_finite():
    # 正样本比负样本低 200 分：损失约等于 200，不能溢出成 inf
    table = torch.tensor([[-100.0, 100.0], [100.0, -100.0]])
    users = torch.tensor([0, 1])
    behind = bpr_loss(TableScorer(table), users, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert torch.isfinite(behind), "分数差很大时损失出现了 inf 或 nan"
    assert_close(behind, torch.tensor(200.0), "正样本落后 200 分时的损失", atol=1e-3)
    ahead = bpr_loss(TableScorer(table), users, torch.tensor([1, 0]), torch.tensor([0, 1]))
    assert_close(ahead, torch.tensor(0.0), "正样本领先 200 分时的损失", atol=1e-6)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_matches_reference(seed):
    table, users, pos_items, neg_items = random_case(seed, batch=16)
    scorer = TableScorer(table)
    assert_close(
        bpr_loss(scorer, users, pos_items, neg_items),
        reference_loss(scorer, users, pos_items, neg_items),
        "损失",
    )
