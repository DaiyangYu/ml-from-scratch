"""Transformer Block 的参考实现。手写时不要看；写完后用 `uv run drill diff transformer_block --ref` 对照。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    # q: [B, heads, Lq, d_k]；k, v: [B, heads, Lk, d_k]
    # 除以 sqrt(d_k)，d_k 是每个头的维度，不是 d_model；不缩放的话点积随维度变大，softmax 会趋于饱和
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # [B, heads, Lq, Lk]
    # mask 必须加在 softmax 之前：填成负无穷，归一化后这些位置的权重才是 0，其余位置的权重之和仍为 1
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # 在键的维度上归一化
    return weights @ v


class MultiHead(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _split(self, x):  # [B, L, d_model] -> [B, heads, L, d_k]
        B, L, _ = x.shape
        # 先把最后一维拆成 [heads, d_k]，再把 heads 换到前面；直接 view 成 [B, heads, L, d_k] 会把不同位置的数据混在一起
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        out = attention(self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x)), mask)
        # 合并各头是拆分的逆过程：先把 heads 换回后面，再合并最后两维
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.Wo(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=256):
        super().__init__()
        self.attn = MultiHead(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        # Pre-LN：LayerNorm 在子层之前，残差支路上的 x 不经过 LayerNorm。
        # 原论文是 Post-LN：x = ln(x + sublayer(x))
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
