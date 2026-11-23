"""ESMM 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from esmm import ESMM, esmm_loss
from esmm_ref import ESMM as Reference
from esmm_ref import esmm_loss as reference_loss


def tower_by_hand(tower, x):
    """逐样本计算一个塔的输出：sigmoid(W2 · relu(W1 x + b1) + b2)。"""
    first, second = tower[0], tower[2]
    rows = []
    for sample in x:
        hidden = torch.clamp(first.weight @ sample + first.bias, min=0)
        logit = second.weight[0] @ hidden + second.bias[0]
        rows.append(1 / (1 + torch.exp(-logit)))
    return torch.stack(rows).detach()


def bce_by_hand(p, y):
    """二元交叉熵，对 batch 取平均。"""
    y = y.double()
    p = p.double()
    return (-(y * torch.log(p) + (1 - y) * torch.log(1 - p))).mean().float()


def check_forward(model, x, expected):
    outputs = model(x)
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 3, "forward 应返回 (p_ctr, p_cvr, p_ctcvr) 三项"
    for name, mine, theirs in zip(["p_ctr", "p_cvr", "p_ctcvr"], outputs, expected):
        assert_close(mine, theirs, name)


def test_parameters_follow_readme():
    paired(ESMM, Reference, 7, hidden=5)


@pytest.mark.parametrize("batch, in_dim, hidden", [(6, 8, 10), (1, 3, 4), (4, 1, 2)])
def test_forward_matches_hand_computation(batch, in_dim, hidden):
    torch.manual_seed(0)
    model = randomize(ESMM(in_dim, hidden=hidden))
    x = torch.randn(batch, in_dim)
    p_ctr = tower_by_hand(model.ctr_tower, x)
    p_cvr = tower_by_hand(model.cvr_tower, x)
    check_forward(model, x, (p_ctr, p_cvr, p_ctr * p_cvr))


def test_outputs_are_probabilities():
    torch.manual_seed(1)
    model = randomize(ESMM(6, hidden=8), seed=1, scale=2.0)
    for name, p in zip(["p_ctr", "p_cvr", "p_ctcvr"], model(torch.randn(32, 6) * 3)):
        assert p.dim() == 1, f"{name} 的形状应为 [B]，实际为 {tuple(p.shape)}"
        assert (p >= 0).all() and (p <= 1).all(), f"{name} 应在 0 到 1 之间"


def test_ctcvr_never_exceeds_ctr_or_cvr():
    torch.manual_seed(2)
    model = randomize(ESMM(5, hidden=6), seed=2)
    p_ctr, p_cvr, p_ctcvr = model(torch.randn(16, 5))
    assert (p_ctcvr <= p_ctr + 1e-6).all(), "pCTCVR 不应大于 pCTR"
    assert (p_ctcvr <= p_cvr + 1e-6).all(), "pCTCVR 不应大于 pCVR"


def test_two_towers_are_independent():
    torch.manual_seed(3)
    model = randomize(ESMM(5, hidden=6), seed=3)
    x = torch.randn(8, 5)
    p_ctr, p_cvr, _ = model(x)
    assert not torch.allclose(p_ctr, p_cvr), "两个塔的参数不同，pCTR 与 pCVR 不应相同"
    assert_close(p_cvr, tower_by_hand(model.cvr_tower, x), "p_cvr")


LOSS_CASES = [
    # p_ctr, p_ctcvr, click, conversion
    ([0.9, 0.2, 0.6, 0.3], [0.5, 0.05, 0.1, 0.2], [1, 0, 1, 0], [1, 0, 0, 0]),
    ([0.7, 0.4, 0.8], [0.3, 0.3, 0.6], [1, 1, 1], [0, 1, 1]),
    ([0.15], [0.02], [0], [0]),
    ([0.5, 0.25, 0.75, 0.6, 0.35], [0.1, 0.2, 0.7, 0.05, 0.3], [0, 1, 1, 0, 1], [0, 0, 1, 0, 1]),
]


@pytest.mark.parametrize("p_ctr, p_ctcvr, click, conversion", LOSS_CASES)
@pytest.mark.parametrize("label_type", [torch.long, torch.float32, torch.bool])
def test_loss_matches_hand_computation(p_ctr, p_ctcvr, click, conversion, label_type):
    p_ctr, p_ctcvr = torch.tensor(p_ctr), torch.tensor(p_ctcvr)
    click = torch.tensor(click).to(label_type)
    conversion = torch.tensor(conversion).to(label_type)
    expected = bce_by_hand(p_ctr, click) + bce_by_hand(p_ctcvr, conversion)
    loss = esmm_loss(p_ctr, p_ctcvr, click, conversion)
    assert_close(loss, expected, f"标签类型为 {label_type} 时的损失")


def test_loss_is_scalar_and_differentiable():
    torch.manual_seed(4)
    model = randomize(ESMM(5, hidden=6), seed=4)
    p_ctr, p_cvr, p_ctcvr = model(torch.randn(10, 5))
    click = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    conversion = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    loss = esmm_loss(p_ctr, p_ctcvr, click, conversion)
    assert loss.dim() == 0, "损失应为标量"
    loss.backward()
    cvr_grad = model.cvr_tower[0].weight.grad
    ctr_grad = model.ctr_tower[0].weight.grad
    assert cvr_grad is not None and cvr_grad.abs().sum() > 0, "CVR 塔应当通过 pCTCVR 这一项得到梯度"
    assert ctr_grad is not None and ctr_grad.abs().sum() > 0, "CTR 塔应当得到梯度"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(ESMM, Reference, 12, hidden=16, seed=seed)
    x = torch.randn(9, 12)
    expected = [p.detach() for p in reference(x)]
    check_forward(model, x, expected)
    click = torch.randint(0, 2, (9,))
    conversion = click * torch.randint(0, 2, (9,))
    p_ctr, _, p_ctcvr = model(x)
    assert_close(
        esmm_loss(p_ctr, p_ctcvr, click, conversion),
        reference_loss(expected[0], expected[2], click, conversion),
        "损失",
    )
