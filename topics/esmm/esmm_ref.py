"""ESMM 的参考实现。手写时不要看；写完后用 `uv run drill diff esmm --ref` 对照。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_tower(in_dim, hidden):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1), nn.Sigmoid())


class ESMM(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        # 两个塔结构相同、参数各自独立；完整模型里它们共享底层的 Embedding
        self.ctr_tower = make_tower(in_dim, hidden)
        self.cvr_tower = make_tower(in_dim, hidden)

    def forward(self, x):  # x: [B, in_dim] 共享表示
        p_ctr = self.ctr_tower(x).squeeze(-1)  # [B]
        p_cvr = self.cvr_tower(x).squeeze(-1)  # [B]，没有直接的监督信号
        # 核心公式：pCTCVR = pCTR × pCVR
        return p_ctr, p_cvr, p_ctr * p_cvr


def esmm_loss(p_ctr, p_ctcvr, click, conversion):
    # 两项都在全部曝光样本上计算：CTR 用点击标签，CTCVR 用“点击且转化”标签。
    # 不能给 pCVR 单独加一项只用点击样本的损失，那样就退回了有样本选择偏差的做法。
    ctr_loss = F.binary_cross_entropy(p_ctr, click.float())
    ctcvr_loss = F.binary_cross_entropy(p_ctcvr, conversion.float())
    return ctr_loss + ctcvr_loss
