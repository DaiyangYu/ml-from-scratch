"""AUGRU 的测试。"""

import pytest
import torch
import torch.nn as nn

from augru import AUGRUCell
from augru_ref import AUGRUCell as Reference
from drill.testing import assert_close, paired, randomize

SHAPES = [(5, 7, 4), (6, 6, 3), (1, 3, 1), (9, 2, 5)]  # input_dim, hidden_dim, batch


def with_official_weights(input_dim, hidden_dim, seed=0):
    """建立官方 nn.GRUCell 和装入同一组权重的待测模块。"""
    official = randomize(nn.GRUCell(input_dim, hidden_dim), seed)
    model = AUGRUCell(input_dim, hidden_dim)
    model.load_state_dict(official.state_dict())
    return model, official


def test_parameters_follow_readme():
    paired(AUGRUCell, Reference, 5, 7)


@pytest.mark.parametrize("input_dim, hidden_dim, batch", SHAPES)
def test_score_one_is_plain_gru(input_dim, hidden_dim, batch):
    torch.manual_seed(0)
    model, official = with_official_weights(input_dim, hidden_dim)
    x, h = torch.randn(batch, input_dim), torch.randn(batch, hidden_dim)
    with torch.no_grad():
        assert_close(model(x, h, torch.ones(batch, 1)), official(x, h),
                     "注意力分数为 1 时的输出（应与 nn.GRUCell 相同）")


@pytest.mark.parametrize("input_dim, hidden_dim, batch", SHAPES)
def test_score_zero_keeps_state(input_dim, hidden_dim, batch):
    torch.manual_seed(1)
    model, _ = with_official_weights(input_dim, hidden_dim, seed=1)
    x, h = torch.randn(batch, input_dim), torch.randn(batch, hidden_dim)
    with torch.no_grad():
        assert_close(model(x, h, torch.zeros(batch, 1)), h,
                     "注意力分数为 0 时的输出（应等于旧状态）")


@pytest.mark.parametrize("input_dim, hidden_dim, batch", SHAPES)
def test_general_score(input_dim, hidden_dim, batch):
    # 按论文公式，新状态相对旧状态的变化量正好是普通 GRU 变化量的 a 倍
    torch.manual_seed(2)
    model, official = with_official_weights(input_dim, hidden_dim, seed=2)
    x, h = torch.randn(batch, input_dim), torch.randn(batch, hidden_dim)
    att_score = torch.rand(batch, 1)
    with torch.no_grad():
        expected = h + att_score * (official(x, h) - h)
        assert_close(model(x, h, att_score), expected, "一般注意力分数下的输出")


def test_each_sample_uses_its_own_score():
    torch.manual_seed(3)
    model, official = with_official_weights(4, 6, seed=3)
    x, h = torch.randn(3, 4), torch.randn(3, 6)
    att_score = torch.tensor([[0.0], [1.0], [0.25]])
    with torch.no_grad():
        expected = h + att_score * (official(x, h) - h)
        assert_close(model(x, h, att_score), expected, "同一批内注意力分数不同时的输出")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(AUGRUCell, Reference, 6, 8, seed=seed)
    x, h, att_score = torch.randn(5, 6), torch.randn(5, 8), torch.rand(5, 1)
    with torch.no_grad():
        assert_close(model(x, h, att_score), reference(x, h, att_score))
