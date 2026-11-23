"""FM 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from fm import FM
from fm_ref import FM as Reference


def by_definition(model, feat_ids):
    """按定义逐样本计算：w0 + Σ w_i + Σ_{i<j} <v_i, v_j>。"""
    outputs = []
    for ids in feat_ids.tolist():
        y = model.w0.detach().clone().squeeze()
        for i in ids:
            y = y + model.linear.weight[i, 0]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                y = y + torch.dot(model.emb.weight[ids[a]], model.emb.weight[ids[b]])
        outputs.append(y)
    return torch.stack(outputs).unsqueeze(-1).detach()


def test_parameters_follow_readme():
    paired(FM, Reference, 50, dim=8)


def test_w0_starts_at_zero():
    assert FM(10, dim=4).w0.detach().abs().sum().item() == 0


@pytest.mark.parametrize("batch, n, dim", [(4, 5, 8), (1, 2, 3), (3, 9, 1), (2, 1, 4)])
def test_matches_definition(batch, n, dim):
    torch.manual_seed(0)
    model = randomize(FM(30, dim=dim))
    feat_ids = torch.randint(0, 30, (batch, n))
    assert_close(model(feat_ids), by_definition(model, feat_ids))


def test_repeated_feature_in_one_sample():
    model = randomize(FM(30, dim=6), seed=1)
    feat_ids = torch.tensor([[3, 3, 7, 3]])
    assert_close(model(feat_ids), by_definition(model, feat_ids))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(FM, Reference, 100, dim=16, seed=seed)
    feat_ids = torch.randint(0, 100, (8, 12))
    assert_close(model(feat_ids), reference(feat_ids))
