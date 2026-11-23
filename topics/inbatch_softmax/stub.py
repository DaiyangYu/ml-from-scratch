"""批内负采样 Softmax 损失。要求见同目录下的 README.md。"""

import torch
import torch.nn.functional as F


def inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.05, item_log_probs=None):
    # user_vecs, item_vecs: [B, d]; item_log_probs: [B] 或 None -> 标量
    raise NotImplementedError
