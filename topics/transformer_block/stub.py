"""Transformer Block：注意力、多头注意力、编码器块。要求见同目录下的 README.md。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):  # [B, heads, Lq, d_k], [B, heads, Lk, d_k] x 2 -> [B, heads, Lq, d_k]
    raise NotImplementedError


class MultiHead(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):  # [B, L, d_model] -> [B, L, d_model]
        raise NotImplementedError


class EncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=256):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):  # [B, L, d_model] -> [B, L, d_model]
        raise NotImplementedError
