"""DCN-v2 交叉层的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from dcn_cross import CrossLayerV2, CrossNetV2
from dcn_cross_ref import CrossLayerV2 as ReferenceLayer
from dcn_cross_ref import CrossNetV2 as ReferenceNet


def layer_by_formula(layer, x0, xl):
    """按论文公式逐样本计算一层交叉：x0 ⊙ (W xl + b) + xl。"""
    weight, bias = layer.linear.weight.detach(), layer.linear.bias.detach()
    rows = [x0[s] * (weight @ xl[s] + bias) + xl[s] for s in range(x0.size(0))]
    return torch.stack(rows)


def test_layer_parameters_follow_readme():
    paired(CrossLayerV2, ReferenceLayer, 6)


def test_net_parameters_follow_readme():
    paired(CrossNetV2, ReferenceNet, 6, num_layers=4)


@pytest.mark.parametrize("batch, dim", [(4, 6), (1, 3), (5, 1)])
def test_layer_matches_formula(batch, dim):
    torch.manual_seed(0)
    layer = randomize(CrossLayerV2(dim))
    x0, xl = torch.randn(batch, dim), torch.randn(batch, dim)
    assert_close(layer(x0, xl), layer_by_formula(layer, x0, xl), "单层交叉的输出")


def test_layer_distinguishes_x0_from_xl():
    # x0 与 xl 不同时，交换二者应当得到不同的结果
    torch.manual_seed(1)
    layer = randomize(CrossLayerV2(5), seed=1)
    x0, xl = torch.randn(3, 5), torch.randn(3, 5)
    assert_close(layer(x0, xl), layer_by_formula(layer, x0, xl), "单层交叉的输出")
    assert_close(layer(xl, x0), layer_by_formula(layer, xl, x0), "交换两个输入后单层交叉的输出")


def test_layer_with_zero_x0_returns_xl():
    layer = randomize(CrossLayerV2(4), seed=2)
    xl = torch.randn(3, 4)
    assert_close(layer(torch.zeros(3, 4), xl), xl, "x0 全为 0 时的输出（应当等于 xl）")


@pytest.mark.parametrize("num_layers", [1, 2, 3, 5])
def test_net_matches_layer_by_layer_formula(num_layers):
    torch.manual_seed(num_layers)
    net = randomize(CrossNetV2(6, num_layers=num_layers), seed=num_layers, scale=0.3)
    assert len(net.layers) == num_layers, f"layers 应包含 {num_layers} 层，实际为 {len(net.layers)} 层"
    x0 = torch.randn(4, 6)
    expected = x0
    for layer in net.layers:
        expected = layer_by_formula(layer, x0, expected)
    assert_close(net(x0), expected, f"{num_layers} 层交叉网络的输出", atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_layer_matches_reference(seed):
    torch.manual_seed(seed)
    layer, reference = paired(CrossLayerV2, ReferenceLayer, 8, seed=seed)
    x0, xl = torch.randn(5, 8), torch.randn(5, 8)
    assert_close(layer(x0, xl), reference(x0, xl), "单层交叉的输出")


@pytest.mark.parametrize("dim, num_layers, seed", [(8, 3, 0), (4, 1, 1), (10, 4, 2)])
def test_net_matches_reference(dim, num_layers, seed):
    torch.manual_seed(seed)
    net, reference = paired(CrossNetV2, ReferenceNet, dim, num_layers=num_layers, seed=seed)
    x0 = torch.randn(6, dim) * 0.5
    assert_close(net(x0), reference(x0), "交叉网络的输出", atol=1e-4)
