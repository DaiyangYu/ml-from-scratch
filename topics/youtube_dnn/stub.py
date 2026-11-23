"""YouTubeDNN 召回。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class YouTubeDNN(nn.Module):
    def __init__(self, num_items, dim=64, hidden=128):
        super().__init__()
        raise NotImplementedError

    def forward(self, hist_ids, hist_mask, cand_ids):  # [B, L], [B, L], [B, K] -> [B, K]
        raise NotImplementedError
