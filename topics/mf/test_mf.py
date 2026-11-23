"""MF 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from mf import MF
from mf_ref import MF as Reference


def by_definition(model, user_ids, item_ids):
    """按定义逐样本计算用户向量与物品向量的点积。"""
    scores = [
        torch.dot(model.user_emb.weight[u], model.item_emb.weight[i])
        for u, i in zip(user_ids.tolist(), item_ids.tolist())
    ]
    return torch.stack(scores).detach()


def test_parameters_follow_readme():
    paired(MF, Reference, 20, 30, dim=8)


def test_hand_computed_values():
    model = MF(2, 3, dim=2)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.item_emb.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
    scores = model(torch.tensor([0, 1, 1, 0]), torch.tensor([0, 2, 1, 2]))
    assert_close(scores, torch.tensor([1.0, 14.0, 4.0, 6.0]), "预测分")


@pytest.mark.parametrize("batch, dim", [(6, 8), (1, 3), (5, 1)])
def test_matches_definition(batch, dim):
    torch.manual_seed(0)
    model = randomize(MF(20, 30, dim=dim))
    user_ids = torch.randint(0, 20, (batch,))
    item_ids = torch.randint(0, 30, (batch,))
    assert_close(model(user_ids, item_ids), by_definition(model, user_ids, item_ids), "预测分")


def test_same_user_with_different_items():
    model = randomize(MF(20, 30, dim=4), seed=1)
    user_ids = torch.tensor([7, 7, 7])
    item_ids = torch.tensor([0, 29, 13])
    assert_close(model(user_ids, item_ids), by_definition(model, user_ids, item_ids), "预测分")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(MF, Reference, 40, 60, dim=16, seed=seed)
    user_ids = torch.randint(0, 40, (9,))
    item_ids = torch.randint(0, 60, (9,))
    assert_close(model(user_ids, item_ids), reference(user_ids, item_ids), "预测分")
