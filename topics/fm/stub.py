"""FM：因子分解机。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_features, dim=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, feat_ids):  # [B, n] -> [B, 1]
        raise NotImplementedError
