"""DSSM 双塔的参考实现。手写时不要看；写完后用 `uv run drill diff dssm --ref` 对照。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_tower(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class DSSM(nn.Module):
    def __init__(self, user_dim, item_dim, out_dim=64, hidden=128):
        super().__init__()
        # 两个塔的输入维度可以不同，输出维度必须相同，否则无法做点积
        self.user_tower = make_tower(user_dim, hidden, out_dim)
        self.item_tower = make_tower(item_dim, hidden, out_dim)

    def encode_user(self, user_feats):  # [B, user_dim] -> [B, out_dim]
        # L2 归一化不能漏：归一化之后的点积才是余弦相似度
        return F.normalize(self.user_tower(user_feats), dim=-1)

    def encode_item(self, item_feats):  # [B, item_dim] -> [B, out_dim]
        return F.normalize(self.item_tower(item_feats), dim=-1)

    def forward(self, user_feats, item_feats):
        u = self.encode_user(user_feats)
        i = self.encode_item(item_feats)
        # 两个塔直到最后一步才相遇，所以物品向量可以离线算好建索引，线上只算用户向量
        return (u * i).sum(dim=-1)  # [B]
