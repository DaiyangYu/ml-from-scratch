"""BPR 损失。要求见同目录下的 README.md。"""

import torch
import torch.nn.functional as F


def bpr_loss(model, users, pos_items, neg_items):  # 三个 id 张量均为 [B] -> 标量
    raise NotImplementedError
