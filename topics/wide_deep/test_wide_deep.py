"""Wide & Deep 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from wide_deep import WideAndDeep
from wide_deep_ref import WideAndDeep as Reference


def by_structure(model, wide_x, feat_ids):
    """按结构逐样本计算：Wide 的线性输出 + Deep 的多层感知机输出。"""
    first, second = model.deep[0], model.deep[2]
    rows = []
    for sample_wide, ids in zip(wide_x, feat_ids.tolist()):
        wide_logit = model.wide.weight[0] @ sample_wide + model.wide.bias[0]
        deep_in = torch.cat([model.emb.weight[i] for i in ids])
        hidden = torch.clamp(first.weight @ deep_in + first.bias, min=0)
        deep_logit = second.weight[0] @ hidden + second.bias[0]
        rows.append(wide_logit + deep_logit)
    return torch.stack(rows).unsqueeze(-1).detach()


def test_parameters_follow_readme():
    # 特征 id 总数（50）与特征域个数（4）不相等
    paired(WideAndDeep, Reference, 9, 50, 4, emb_dim=6, hidden=10)


def test_deep_input_dim_uses_number_of_fields():
    model = WideAndDeep(5, 1000, 8, emb_dim=4, hidden=12)
    assert tuple(model.emb.weight.shape) == (1000, 4), "Embedding 表的行数应为特征 id 总数"
    assert model.deep[0].in_features == 8 * 4, "Deep 第一层的输入维度应为 特征域个数 × embedding 维度"


@pytest.mark.parametrize(
    "batch, wide_dim, num_features, n_fields, emb_dim, hidden",
    [
        (4, 9, 50, 4, 6, 10),
        (2, 3, 1000, 8, 4, 12),  # 特征 id 总数远大于特征域个数
        (3, 5, 3, 7, 2, 6),      # 特征域个数大于特征 id 总数
        (1, 1, 10, 1, 3, 4),
    ],
)
def test_matches_structure(batch, wide_dim, num_features, n_fields, emb_dim, hidden):
    torch.manual_seed(0)
    model = randomize(WideAndDeep(wide_dim, num_features, n_fields, emb_dim=emb_dim, hidden=hidden))
    wide_x = torch.randn(batch, wide_dim)
    feat_ids = torch.randint(0, num_features, (batch, n_fields))
    assert_close(model(wide_x, feat_ids), by_structure(model, wide_x, feat_ids))


def test_field_order_matters():
    # 各特征域的 embedding 是拼接的，交换两个域的 id 应当改变输出
    torch.manual_seed(1)
    model = randomize(WideAndDeep(4, 20, 3, emb_dim=5, hidden=8), seed=1)
    wide_x = torch.randn(2, 4)
    feat_ids = torch.tensor([[1, 7, 12], [3, 3, 9]])
    swapped = feat_ids[:, [1, 0, 2]]
    assert_close(model(wide_x, feat_ids), by_structure(model, wide_x, feat_ids))
    assert_close(model(wide_x, swapped), by_structure(model, wide_x, swapped), "交换特征域之后的输出")


def test_output_is_logit_not_probability():
    torch.manual_seed(2)
    model = randomize(WideAndDeep(6, 30, 4, emb_dim=4, hidden=8), seed=2, scale=2.0)
    out = model(torch.randn(64, 6) * 3, torch.randint(0, 30, (64, 4)))
    assert out.min().item() < 0 or out.max().item() > 1, "输出应为 logit，不应过 sigmoid"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(WideAndDeep, Reference, 12, 200, 6, emb_dim=16, hidden=64, seed=seed)
    wide_x = torch.randn(8, 12)
    feat_ids = torch.randint(0, 200, (8, 6))
    assert_close(model(wide_x, feat_ids), reference(wide_x, feat_ids), atol=1e-4)
