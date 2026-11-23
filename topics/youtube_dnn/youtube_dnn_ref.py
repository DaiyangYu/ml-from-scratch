"""YouTubeDNN 召回的参考实现。手写时不要看；写完后用 `uv run drill diff youtube_dnn --ref` 对照。"""

import torch
import torch.nn as nn


class YouTubeDNN(nn.Module):
    def __init__(self, num_items, dim=64, hidden=128):
        super().__init__()
        # 历史物品和候选物品共用这一张表；物品侧没有塔，这一行向量就相当于 softmax 的类别权重
        self.item_emb = nn.Embedding(num_items, dim)
        # 塔的输出维度要等于 dim，才能与物品向量做点积
        self.user_tower = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, hist_ids, hist_mask, cand_ids):
        h = self.item_emb(hist_ids)           # [B, L, dim]
        m = hist_mask.unsqueeze(-1).float()   # [B, L, 1] 转成浮点才能相乘
        # 带掩码的平均池化：分子只累加真实历史，分母是真实历史的个数而不是 L。
        # 历史全是 padding 时分母为 0，用 clamp 把下限设为 1，避免 0 / 0 得到 nan
        avg = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)  # [B, dim]
        u = self.user_tower(avg)              # [B, dim] 用户向量
        cand = self.item_emb(cand_ids)        # [B, K, dim]
        # 用户向量与每个候选做点积。训练时对这 K 个 logits 做交叉熵，就是 sampled softmax
        return (u.unsqueeze(1) * cand).sum(dim=-1)  # [B, K]
