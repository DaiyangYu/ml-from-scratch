"""批内负采样 Softmax 损失的参考实现。手写时不要看；写完后用 `uv run drill diff inbatch_softmax --ref` 对照。"""

import torch
import torch.nn.functional as F


def inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.05, item_log_probs=None):
    # 第 i 行是第 i 个用户对 batch 中全部 B 个物品的打分：[B, B]
    # 温度越小，分数之间的差距被放得越大，模型越关注难分的负样本
    logits = user_vecs @ item_vecs.T / temperature
    if item_log_probs is not None:
        # logQ 纠偏：第 j 列是第 j 个物品，整列减去它的 log p（unsqueeze(0) 让 [B] 对齐到列上）。
        # 对角线上的正样本也一起减。热门物品的 log p 大，减掉之后它作为负样本的打分变低，受到的打压变轻。
        # 修正量不除以温度：它是加在最终 logit 上的。
        logits = logits - item_log_probs.unsqueeze(0)
    # 第 i 个用户的正样本是第 i 个物品，所以标签就是 0, 1, …, B-1（对角线）
    labels = torch.arange(user_vecs.size(0), device=user_vecs.device)
    return F.cross_entropy(logits, labels)  # 按行做 softmax，默认对 batch 取平均
