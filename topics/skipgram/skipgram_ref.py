"""Skip-gram + 负采样的参考实现。手写时不要看；写完后用 `uv run drill diff skipgram --ref` 对照。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim=100):
        super().__init__()
        # 两张表：同一个词作中心词和作上下文词时用不同的向量。训练完通常只保留中心词表
        self.center_emb = nn.Embedding(vocab_size, dim)
        self.context_emb = nn.Embedding(vocab_size, dim)

    def forward(self, center, pos_context, neg_contexts):
        u = self.center_emb(center)             # [B, dim]
        v_pos = self.context_emb(pos_context)   # [B, dim]     上下文词和负样本词都查上下文表
        v_neg = self.context_emb(neg_contexts)  # [B, K, dim]
        pos_score = (u * v_pos).sum(dim=-1)                        # [B]
        neg_score = torch.bmm(v_neg, u.unsqueeze(-1)).squeeze(-1)  # [B, K]
        # 正样本对：希望 σ(score) 接近 1，损失为 -log σ(score)
        pos_loss = -F.logsigmoid(pos_score)                # [B]
        # 负样本对：希望 σ(score) 接近 0，即 σ(-score) 接近 1。这里的负号最容易写反
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=-1)   # [B] 按原论文对 K 个负样本求和，不取平均
        return (pos_loss + neg_loss).mean()                # 对 batch 取平均
