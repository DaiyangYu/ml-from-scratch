"""Transformer Block 的测试：三个部件分别对照 PyTorch 官方实现，再与参考实现对比。"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from drill.testing import assert_close, paired, randomize
from transformer_block import EncoderBlock, MultiHead, attention
from transformer_block_ref import EncoderBlock as ReferenceBlock
from transformer_block_ref import MultiHead as ReferenceMultiHead

MASK_KINDS = ["none", "causal", "keys", "causal+keys"]
CONFIGS = [(16, 4, 3, 6), (8, 1, 2, 5), (8, 8, 1, 4), (12, 3, 4, 7)]  # d_model, num_heads, batch, length


def build_masks(kind, batch, length):
    """返回（本题约定的 mask，官方的 attn_mask，官方的 key_padding_mask）。官方约定是 True 表示屏蔽。"""
    causal = torch.tril(torch.ones(length, length))
    keys = torch.ones(batch, length)
    for b in range(batch):
        keys[b, length - 1 - b % 2:] = 0  # 屏蔽末尾的一到两个键，第 0 个键始终可见
    if kind == "none":
        return None, None, None
    if kind == "causal":
        return causal.view(1, 1, length, length), causal == 0, None
    if kind == "keys":
        return keys.view(batch, 1, 1, length), None, keys == 0
    return causal.view(1, 1, length, length) * keys.view(batch, 1, 1, length), causal == 0, keys == 0


def official_attention(multi_head, d_model, num_heads):
    """把 MultiHead 的权重搬进 nn.MultiheadAttention。"""
    official = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
    with torch.no_grad():
        official.in_proj_weight.copy_(torch.cat([multi_head.Wq.weight, multi_head.Wk.weight, multi_head.Wv.weight]))
        official.in_proj_bias.copy_(torch.cat([multi_head.Wq.bias, multi_head.Wk.bias, multi_head.Wv.bias]))
        official.out_proj.weight.copy_(multi_head.Wo.weight)
        official.out_proj.bias.copy_(multi_head.Wo.bias)
    return official.train()  # dropout 为 0，训练模式下结果相同，同时避开推理专用的快速路径


def official_block(block, d_model, num_heads, d_ff):
    """把 EncoderBlock 的权重搬进 Pre-LN 的 nn.TransformerEncoderLayer。"""
    official = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=0.0, activation="relu",
                                          batch_first=True, norm_first=True)
    official.self_attn.load_state_dict(official_attention(block.attn, d_model, num_heads).state_dict())
    official.linear1.load_state_dict(block.ffn[0].state_dict())
    official.linear2.load_state_dict(block.ffn[2].state_dict())
    official.norm1.load_state_dict(block.ln1.state_dict())
    official.norm2.load_state_dict(block.ln2.state_dict())
    return official.train()


# ---------- attention ----------

@pytest.mark.parametrize("kind", MASK_KINDS)
def test_attention_matches_official(kind):
    torch.manual_seed(0)
    batch, heads, length, d_k = 3, 2, 6, 5
    q, k, v = (torch.randn(batch, heads, length, d_k) for _ in range(3))
    mask, _, _ = build_masks(kind, batch, length)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.bool())
    assert_close(attention(q, k, v, mask), expected, f"attention 的输出（mask：{kind}）")


def test_attention_with_different_query_and_key_lengths():
    torch.manual_seed(1)
    q = torch.randn(2, 3, 4, 8)
    k, v = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    mask = torch.ones(2, 1, 4, 7)
    mask[0, 0, :, 5:] = 0
    mask[1, 0, 2, :3] = 0
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool())
    assert_close(attention(q, k, v, mask), expected, "查询与键长度不同时 attention 的输出")


# ---------- MultiHead ----------

def test_multi_head_parameters_follow_readme():
    paired(MultiHead, ReferenceMultiHead, 16, 4)


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("d_model, num_heads, batch, length", CONFIGS)
def test_multi_head_matches_official(kind, d_model, num_heads, batch, length):
    torch.manual_seed(2)
    model = randomize(MultiHead(d_model, num_heads), seed=2)
    official = official_attention(model, d_model, num_heads)
    x = torch.randn(batch, length, d_model)
    mask, attn_mask, key_padding_mask = build_masks(kind, batch, length)
    with torch.no_grad():
        expected, _ = official(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        assert_close(model(x, mask), expected, f"MultiHead 的输出（对照 nn.MultiheadAttention，mask：{kind}）")


def test_masked_key_has_no_influence_on_other_positions():
    torch.manual_seed(3)
    model = randomize(MultiHead(16, 4), seed=3)
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 1, 1, 6)
    mask[:, :, :, 4] = 0  # 所有查询都看不到第 4 个键
    changed = x.clone()
    changed[:, 4] = torch.randn(2, 16)
    others = [0, 1, 2, 3, 5]
    with torch.no_grad():
        assert_close(model(changed, mask)[:, others], model(x, mask)[:, others],
                     "被屏蔽的键改变内容后，其他位置的输出")


# ---------- EncoderBlock ----------

def test_block_parameters_follow_readme():
    paired(EncoderBlock, ReferenceBlock, 16, 4, 32)


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("d_model, num_heads, batch, length", CONFIGS)
def test_block_matches_official(kind, d_model, num_heads, batch, length):
    torch.manual_seed(4)
    d_ff = 2 * d_model + 3
    model = randomize(EncoderBlock(d_model, num_heads, d_ff), seed=4)
    official = official_block(model, d_model, num_heads, d_ff)
    x = torch.randn(batch, length, d_model)
    mask, attn_mask, key_padding_mask = build_masks(kind, batch, length)
    with torch.no_grad():
        expected = official(x, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)
        assert_close(model(x, mask), expected,
                     f"EncoderBlock 的输出（对照 Pre-LN 的 nn.TransformerEncoderLayer，mask：{kind}）", atol=1e-4)


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("seed", [0, 1])
def test_block_matches_reference(kind, seed):
    torch.manual_seed(seed)
    model, reference = paired(EncoderBlock, ReferenceBlock, 32, 4, 64, seed=seed)
    x = torch.randn(3, 9, 32)
    mask, _, _ = build_masks(kind, 3, 9)
    with torch.no_grad():
        assert_close(model(x, mask), reference(x, mask), f"EncoderBlock 的输出（mask：{kind}）", atol=1e-4)


def test_default_constructor_arguments():
    model, reference = paired(EncoderBlock, ReferenceBlock)
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        assert_close(model(x), reference(x), "默认构造参数下 EncoderBlock 的输出", atol=1e-4)
