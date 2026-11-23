"""SASRec：自注意力序列推荐。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn

from sasrec_given import EncoderBlock  # 现成的 Transformer Block，不需要重写


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, dim=64, num_heads=4, num_layers=2):
        super().__init__()
        raise NotImplementedError

    def forward(self, seq):  # [B, L] -> [B, L, dim]
        raise NotImplementedError

    def score(self, seq, cand_items):  # [B, L], [B, K] -> [B, K]
        raise NotImplementedError
