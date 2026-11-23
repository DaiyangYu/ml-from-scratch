"""AUGRU：带注意力更新门的 GRU 单元。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, h, att_score):  # [B, input_dim], [B, hidden_dim], [B, 1] -> [B, hidden_dim]
        raise NotImplementedError
