"""Skip-gram + 负采样。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim=100):
        super().__init__()
        raise NotImplementedError

    def forward(self, center, pos_context, neg_contexts):  # [B], [B], [B, K] -> 标量
        raise NotImplementedError
