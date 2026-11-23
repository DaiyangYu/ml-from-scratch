"""Wide & Deep 的参考实现。手写时不要看；写完后用 `uv run drill diff wide_deep --ref` 对照。"""

import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, num_features, n_fields, emb_dim=16, hidden=64):
        super().__init__()
        # Wide：一层线性变换（逻辑回归），负责记忆人工交叉特征
        self.wide = nn.Linear(wide_dim, 1)
        # Deep：Embedding + 多层感知机，负责泛化
        self.emb = nn.Embedding(num_features, emb_dim)  # 表的行数是特征 id 总数
        self.deep = nn.Sequential(
            # 输入维度是“特征域个数 × embedding 维度”，不是“特征 id 总数 × embedding 维度”
            nn.Linear(n_fields * emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, wide_x, feat_ids):
        # wide_x: [B, wide_dim]; feat_ids: [B, n_fields]
        deep_in = self.emb(feat_ids).flatten(1)  # [B, n_fields, emb_dim] 拼平成 [B, n_fields * emb_dim]
        # 两部分的 logit 相加，sigmoid 留给外层的损失函数；两部分一起训练，不是分别训练再融合
        return self.wide(wide_x) + self.deep(deep_in)  # [B, 1]
