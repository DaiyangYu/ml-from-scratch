"""MF：矩阵分解。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, dim=32):
        super().__init__()
        raise NotImplementedError

    def forward(self, user_ids, item_ids):  # [B], [B] -> [B]
        raise NotImplementedError
