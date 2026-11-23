"""DeepFM。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_features, n_fields, dim=16, hidden=64):
        super().__init__()
        raise NotImplementedError

    def forward(self, feat_ids):  # [B, n_fields] -> [B, 1]
        raise NotImplementedError
