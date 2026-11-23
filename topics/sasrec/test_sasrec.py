"""SASRec 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from sasrec import SASRec
from sasrec_ref import SASRec as Reference

# num_items, max_len, dim, num_heads, num_layers, batch, length
CONFIGS = [(30, 10, 16, 2, 2, 4, 10), (30, 12, 32, 4, 1, 3, 7), (15, 6, 8, 1, 3, 5, 6)]


def random_sequences(batch, length, num_items, seed=0, pad=True):
    """随机行为序列。pad=True 时第 b 个样本的左侧有 b 个 padding（最后一个位置始终是真实 item）。"""
    generator = torch.Generator().manual_seed(seed)
    seq = torch.randint(1, num_items + 1, (batch, length), generator=generator)
    if pad:
        for b in range(batch):
            seq[b, : b % length] = 0
    return seq


def build(num_items=20, max_len=10, dim=16, num_heads=2, num_layers=2, seed=0):
    return randomize(SASRec(num_items, max_len=max_len, dim=dim, num_heads=num_heads, num_layers=num_layers), seed).eval()


def test_parameters_follow_readme():
    paired(SASRec, Reference, 20, max_len=10, dim=16, num_heads=2, num_layers=3)


def test_default_constructor_arguments():
    paired(SASRec, Reference, 20)


def test_without_blocks_matches_definition():
    # 没有 Block 时，输出就是「item 向量加位置向量，padding 位置置零，再过 LayerNorm」
    model = build(max_len=8, dim=12, num_layers=0)
    seq = random_sequences(4, 6, 20)
    with torch.no_grad():
        hidden = model.item_emb.weight[seq] + model.pos_emb.weight[:6]
        hidden = hidden * (seq != 0).unsqueeze(-1)
        assert_close(model(seq), model.ln(hidden), "没有 Block 时的输出")


def test_output_at_padding_positions():
    # padding 位置的隐状态在最后一个 Block 之后被置零，零向量过 LayerNorm 得到的就是它的偏置
    model = build()
    seq = random_sequences(5, 8, 20)
    with torch.no_grad():
        output = model(seq)
        padding = seq == 0
        assert padding.any()
        expected = model.ln.bias.expand(int(padding.sum()), -1)
        assert_close(output[padding], expected, "padding 位置的输出")


def test_no_nan_with_left_padding():
    model = build()
    seq = random_sequences(6, 8, 20)
    seq[5, :7] = 0  # 只剩最后一个真实 item
    with torch.no_grad():
        output = model(seq)
    assert tuple(output.shape) == (6, 8, 16), f"输出的形状应为 (6, 8, 16)，实际为 {tuple(output.shape)}"
    assert not output.isnan().any(), "左侧有 padding 时输出中出现了 NaN"


@pytest.mark.parametrize("t", [0, 3, 6])
def test_output_does_not_depend_on_later_items(t):
    model = build()
    seq = random_sequences(3, 8, 20, pad=False)
    changed = seq.clone()
    changed[:, t + 1:] = changed[:, t + 1:] % 20 + 1  # 把位置 t 之后的 item 全部换成别的 item
    with torch.no_grad():
        before, after = model(seq), model(changed)
    assert_close(after[:, : t + 1], before[:, : t + 1], f"修改位置 {t} 之后的 item 时，位置 {t} 及之前的输出")
    difference = (after[:, t + 1:] - before[:, t + 1:]).abs().amax(dim=-1)
    assert (difference > 1e-4).all(), f"修改位置 {t} 之后的 item 时，这些位置自身的输出应当改变"


def test_last_position_depends_on_earlier_items():
    model = build()
    seq = random_sequences(3, 8, 20, pad=False)
    changed = seq.clone()
    changed[:, 0] = changed[:, 0] % 20 + 1
    with torch.no_grad():
        difference = (model(changed)[:, -1] - model(seq)[:, -1]).abs().amax(dim=-1)
    assert (difference > 1e-4).all(), "修改第一个 item 后，最后一个位置的输出应当改变"


def test_sequence_shorter_than_max_len():
    model, reference = paired(SASRec, Reference, 20, max_len=10, dim=16, num_heads=2, num_layers=2)
    seq = random_sequences(3, 4, 20)
    with torch.no_grad():
        assert_close(model(seq), reference(seq), "序列长度小于 max_len 时的输出", atol=1e-4)


@pytest.mark.parametrize("pad", [True, False])
def test_score_is_dot_product_with_last_position(pad):
    model = build()
    seq = random_sequences(4, 8, 20, seed=1, pad=pad)
    cand_items = torch.randint(1, 21, (4, 5), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        last = model(seq)[:, -1]  # [B, dim]
        expected = torch.einsum("bd,bkd->bk", last, model.item_emb.weight[cand_items])
        assert_close(model.score(seq, cand_items), expected, "score 的输出", atol=1e-4)


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("num_items, max_len, dim, num_heads, num_layers, batch, length", CONFIGS)
def test_matches_reference(pad, num_items, max_len, dim, num_heads, num_layers, batch, length):
    model, reference = paired(SASRec, Reference, num_items, max_len=max_len, dim=dim,
                              num_heads=num_heads, num_layers=num_layers)
    seq = random_sequences(batch, length, num_items, pad=pad)
    cand_items = torch.randint(1, num_items + 1, (batch, 6), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert_close(model(seq), reference(seq), "forward 的输出", atol=1e-4)
        assert_close(model.score(seq, cand_items), reference.score(seq, cand_items), "score 的输出", atol=1e-4)
