"""FM 的参考实现。手写时不要看；写完后用 `uv run drill diff fm --ref` 对照。"""

import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_features, dim=16):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(num_features, 1)  # 一阶权重 w_i
        self.emb = nn.Embedding(num_features, dim)   # 隐向量 v_i

    def forward(self, feat_ids):  # [B, n]
        linear_part = self.linear(feat_ids).sum(dim=1)   # [B, 1]
        v = self.emb(feat_ids)                           # [B, n, dim]
        # Σ_{i<j} <v_i, v_j> = 0.5 * ((Σ v_i)² − Σ v_i²)，在每个隐维度上分别成立，最后对隐维度求和
        square_of_sum = v.sum(dim=1).pow(2)              # [B, dim] 先求和再平方
        sum_of_square = v.pow(2).sum(dim=1)              # [B, dim] 先平方再求和
        cross = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)  # [B, 1]
        return self.w0 + linear_part + cross
