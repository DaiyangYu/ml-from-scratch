"""DIN Attention 的测试。"""

import pytest
import torch

from din_attention import DINAttention
from din_attention_ref import DINAttention as Reference
from drill.testing import assert_close, paired, randomize

SHAPES = [(4, 5, 8), (1, 1, 3), (3, 7, 1), (2, 6, 16)]  # batch, length, dim


def by_definition(model, target, history, mask=None):
    """逐样本、逐位置计算：每个真实行为的分数乘上它的向量，再求和。"""
    batch, length, dim = history.shape
    output = torch.zeros(batch, dim)
    for b in range(batch):
        for position in range(length):
            if mask is not None and mask[b, position] == 0:
                continue
            t, h = target[b], history[b, position]
            score = model.mlp(torch.cat([t, h, t - h, t * h]).unsqueeze(0)).squeeze()
            output[b] += score * h
    return output


def example_mask(batch, length):
    """前面若干位置是真实行为，其余是 padding；各样本的真实长度不同，至少为 1。"""
    mask = torch.zeros(batch, length)
    for b in range(batch):
        mask[b, : 1 + (b * 2) % length] = 1
    return mask


def test_parameters_follow_readme():
    paired(DINAttention, Reference, 8, hidden=16)


def test_default_hidden_size():
    paired(DINAttention, Reference, 8)


@pytest.mark.parametrize("batch, length, dim", SHAPES)
def test_matches_definition_without_mask(batch, length, dim):
    torch.manual_seed(0)
    model = randomize(DINAttention(dim, hidden=12))
    target, history = torch.randn(batch, dim), torch.randn(batch, length, dim)
    with torch.no_grad():
        assert_close(model(target, history), by_definition(model, target, history), "不带 mask 时的输出")


@pytest.mark.parametrize("batch, length, dim", SHAPES)
def test_matches_definition_with_mask(batch, length, dim):
    torch.manual_seed(1)
    model = randomize(DINAttention(dim, hidden=12), seed=1)
    target, history = torch.randn(batch, dim), torch.randn(batch, length, dim)
    mask = example_mask(batch, length)
    with torch.no_grad():
        assert_close(model(target, history, mask), by_definition(model, target, history, mask), "带 mask 时的输出")


def test_padding_content_has_no_influence():
    torch.manual_seed(2)
    model = randomize(DINAttention(6, hidden=10), seed=2)
    target, history = torch.randn(3, 6), torch.randn(3, 5, 6)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]])
    changed = history.clone()
    changed[mask == 0] = torch.randn(int((mask == 0).sum()), 6) * 10
    with torch.no_grad():
        assert_close(model(target, changed, mask), model(target, history, mask), "padding 位置换成别的向量后的输出")


def test_all_padding_gives_zero_vector():
    torch.manual_seed(3)
    model = randomize(DINAttention(6, hidden=10), seed=3)
    target, history = torch.randn(2, 6), torch.randn(2, 4, 6)
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]])
    with torch.no_grad():
        output = model(target, history, mask)
        assert_close(output[0], torch.zeros(6), "历史全部是 padding 的样本的输出")
        assert_close(output, by_definition(model, target, history, mask))


def test_scores_are_not_normalized():
    # 历史里只有一个 item 时，输出应是「分数 × 该 item 的向量」；若做了 softmax，分数会恒为 1
    torch.manual_seed(4)
    model = randomize(DINAttention(5, hidden=8), seed=4)
    target, history = torch.randn(3, 5), torch.randn(3, 1, 5)
    with torch.no_grad():
        assert_close(model(target, history), by_definition(model, target, history), "历史只有一个 item 时的输出")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(DINAttention, Reference, 16, hidden=32, seed=seed)
    target, history = torch.randn(6, 16), torch.randn(6, 10, 16)
    mask = example_mask(6, 10)
    with torch.no_grad():
        assert_close(model(target, history), reference(target, history), "不带 mask 时的输出")
        assert_close(model(target, history, mask), reference(target, history, mask), "带 mask 时的输出")
