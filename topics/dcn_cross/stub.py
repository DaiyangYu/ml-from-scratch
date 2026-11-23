"""DCN-v2：交叉层与交叉网络。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class CrossLayerV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x0, xl):  # [B, dim], [B, dim] -> [B, dim]
        raise NotImplementedError


class CrossNetV2(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x0):  # [B, dim] -> [B, dim]
        raise NotImplementedError
