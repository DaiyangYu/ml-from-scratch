"""Wide & Deep。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, num_features, n_fields, emb_dim=16, hidden=64):
        super().__init__()
        raise NotImplementedError

    def forward(self, wide_x, feat_ids):  # [B, wide_dim], [B, n_fields] -> [B, 1]
        raise NotImplementedError
