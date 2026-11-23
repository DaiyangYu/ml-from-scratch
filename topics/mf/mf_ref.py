"""MF 的参考实现。手写时不要看；写完后用 `uv run drill diff mf --ref` 对照。"""

import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, dim=32):
        super().__init__()
        # Embedding 就是一张可训练的表，输入 id 取出对应的一行；反向传播时只有被取到的行有梯度
        self.user_emb = nn.Embedding(num_users, dim)  # P: [num_users, dim]
        self.item_emb = nn.Embedding(num_items, dim)  # Q: [num_items, dim]
        # 常见做法是用小方差初始化，避免初始预测分过大（本题不作要求）
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):  # [B], [B]
        u = self.user_emb(user_ids)  # [B, dim]
        i = self.item_emb(item_ids)  # [B, dim]
        # 逐样本点积：对应位置相乘后在向量维度上求和，结果是 [B] 而不是 [B, 1]
        return (u * i).sum(dim=-1)
