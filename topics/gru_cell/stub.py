"""GRU Cell：论文写法与 PyTorch 写法。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class GRUCellPaper(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, h):  # [B, input_dim], [B, hidden_dim] -> [B, hidden_dim]
        raise NotImplementedError


class GRUCellTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, h):  # [B, input_dim], [B, hidden_dim] -> [B, hidden_dim]
        raise NotImplementedError
