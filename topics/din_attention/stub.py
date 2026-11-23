"""DIN Attention：局部激活单元。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class DINAttention(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        raise NotImplementedError

    def forward(self, target, history, mask=None):  # [B, dim], [B, L, dim], [B, L] -> [B, dim]
        raise NotImplementedError
