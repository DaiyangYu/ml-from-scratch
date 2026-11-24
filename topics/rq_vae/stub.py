"""RQ-VAE：残差量化。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, num_levels, codebook_size, dim, beta=0.25):
        super().__init__()
        raise NotImplementedError

    def forward(self, z):  # [B, dim] -> z_q [B, dim], loss 标量, codes [B, num_levels]
        raise NotImplementedError
