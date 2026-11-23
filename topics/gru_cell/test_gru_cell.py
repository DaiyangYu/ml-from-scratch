"""GRU Cell 的测试。"""

import pytest
import torch
import torch.nn as nn

from drill.testing import assert_close, paired, randomize
from gru_cell import GRUCellPaper, GRUCellTorch
from gru_cell_ref import GRUCellPaper as ReferencePaper
from gru_cell_ref import GRUCellTorch as ReferenceTorch

SHAPES = [(5, 7, 4), (6, 6, 3), (1, 3, 1), (9, 2, 5)]  # input_dim, hidden_dim, batch


def by_paper_formula(model, x, h):
    """把每个 Linear 的权重拆成作用于输入和作用于隐状态的两块，按论文公式计算。"""
    input_dim = x.size(-1)

    def parts(linear):
        weight = linear.weight.detach()
        return weight[:, :input_dim], weight[:, input_dim:], linear.bias.detach()

    wz, uz, bz = parts(model.Wz)
    wr, ur, br = parts(model.Wr)
    wh, uh, bh = parts(model.Wh)
    z = torch.sigmoid(x @ wz.T + h @ uz.T + bz)
    r = torch.sigmoid(x @ wr.T + h @ ur.T + br)
    candidate = torch.tanh(x @ wh.T + (r * h) @ uh.T + bh)
    return z * h + (1 - z) * candidate


def with_official_weights(input_dim, hidden_dim, seed=0):
    """建立官方 nn.GRUCell 和装入同一组权重的待测模块。"""
    official = randomize(nn.GRUCell(input_dim, hidden_dim), seed)
    model = GRUCellTorch(input_dim, hidden_dim)
    model.load_state_dict(official.state_dict())
    return model, official


def test_paper_parameters_follow_readme():
    paired(GRUCellPaper, ReferencePaper, 5, 7)


def test_torch_parameters_follow_readme():
    paired(GRUCellTorch, ReferenceTorch, 5, 7)


@pytest.mark.parametrize("input_dim, hidden_dim, batch", SHAPES)
def test_paper_matches_formula(input_dim, hidden_dim, batch):
    torch.manual_seed(0)
    model = randomize(GRUCellPaper(input_dim, hidden_dim))
    x, h = torch.randn(batch, input_dim), torch.randn(batch, hidden_dim)
    with torch.no_grad():
        assert_close(model(x, h), by_paper_formula(model, x, h), "GRUCellPaper 的输出")


@pytest.mark.parametrize("input_dim, hidden_dim, batch", SHAPES)
def test_torch_matches_official(input_dim, hidden_dim, batch):
    torch.manual_seed(0)
    model, official = with_official_weights(input_dim, hidden_dim)
    x, h = torch.randn(batch, input_dim), torch.randn(batch, hidden_dim)
    with torch.no_grad():
        assert_close(model(x, h), official(x, h), "GRUCellTorch 的输出（对照 nn.GRUCell）")


def test_torch_with_zero_hidden_state():
    torch.manual_seed(1)
    model, official = with_official_weights(4, 6, seed=1)
    x, h = torch.randn(3, 4), torch.zeros(3, 6)
    with torch.no_grad():
        assert_close(model(x, h), official(x, h), "旧状态为 0 时 GRUCellTorch 的输出")


def test_torch_unrolled_matches_nn_gru():
    torch.manual_seed(2)
    input_dim, hidden_dim, batch, length = 4, 5, 3, 6
    gru = randomize(nn.GRU(input_dim, hidden_dim, batch_first=True), seed=2)
    model = GRUCellTorch(input_dim, hidden_dim)
    model.load_state_dict({
        "weight_ih": gru.weight_ih_l0, "weight_hh": gru.weight_hh_l0,
        "bias_ih": gru.bias_ih_l0, "bias_hh": gru.bias_hh_l0,
    })
    x = torch.randn(batch, length, input_dim)
    with torch.no_grad():
        h, steps = torch.zeros(batch, hidden_dim), []
        for t in range(length):
            h = model(x[:, t], h)
            steps.append(h)
        expected, _ = gru(x)
        assert_close(torch.stack(steps, dim=1), expected, "沿时间步展开后的输出（对照 nn.GRU）")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paper_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(GRUCellPaper, ReferencePaper, 6, 8, seed=seed)
    x, h = torch.randn(5, 6), torch.randn(5, 8)
    with torch.no_grad():
        assert_close(model(x, h), reference(x, h), "GRUCellPaper 的输出")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(GRUCellTorch, ReferenceTorch, 6, 8, seed=seed)
    x, h = torch.randn(5, 6), torch.randn(5, 8)
    with torch.no_grad():
        assert_close(model(x, h), reference(x, h), "GRUCellTorch 的输出")
