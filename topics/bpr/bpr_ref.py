"""BPR 损失的参考实现。手写时不要看；写完后用 `uv run drill diff bpr --ref` 对照。"""

import torch
import torch.nn.functional as F


def bpr_loss(model, users, pos_items, neg_items):  # 均为 [B]
    pos_score = model(users, pos_items)  # [B] 同一批用户对正样本的打分
    neg_score = model(users, neg_items)  # [B] 同一批用户对负样本的打分
    # 最大化 log σ(正分 − 负分)，取负号变成要最小化的损失。
    # 方向不能反：写成 负分 − 正分，模型会学着把负样本排到前面。
    # 用 logsigmoid 而不是 log(sigmoid(·))：分数差是很大的负数时后者会得到 log(0) = -inf。
    return -F.logsigmoid(pos_score - neg_score).mean()
