"""ESMM：全空间多任务模型。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMM(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):  # [B, in_dim] -> (p_ctr, p_cvr, p_ctcvr)，各为 [B]
        raise NotImplementedError


def esmm_loss(p_ctr, p_ctcvr, click, conversion):  # 均为 [B] -> 标量
    raise NotImplementedError
