"""DSSM 双塔。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM(nn.Module):
    def __init__(self, user_dim, item_dim, out_dim=64, hidden=128):
        super().__init__()
        raise NotImplementedError

    def encode_user(self, user_feats):  # [B, user_dim] -> [B, out_dim]
        raise NotImplementedError

    def encode_item(self, item_feats):  # [B, item_dim] -> [B, out_dim]
        raise NotImplementedError

    def forward(self, user_feats, item_feats):  # -> [B]
        raise NotImplementedError
