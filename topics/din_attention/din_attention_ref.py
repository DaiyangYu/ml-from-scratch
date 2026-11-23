"""DIN Attention 的参考实现。手写时不要看；写完后用 `uv run drill diff din_attention --ref` 对照。"""

import torch
import torch.nn as nn


class DINAttention(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        # 打分用的是多层感知机，不是点积：这是它与 Transformer 注意力最常被问到的区别
        self.mlp = nn.Sequential(nn.Linear(4 * dim, hidden), nn.PReLU(), nn.Linear(hidden, 1))

    def forward(self, target, history, mask=None):
        # target: [B, dim]; history: [B, L, dim]; mask: [B, L]
        B, L, d = history.shape
        t = target.unsqueeze(1).expand(B, L, d)  # 把目标复制到每个历史位置上
        # 四路拼接：目标、历史、目标减历史、目标乘历史
        feats = torch.cat([t, history, t - history, t * history], dim=-1)  # [B, L, 4 * dim]
        scores = self.mlp(feats).squeeze(-1)  # [B, L]
        # 不做 softmax，所以 padding 位置的分数直接置 0；如果是 softmax 版，才是填负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0.0)
        # 原论文不做归一化：分数之和的大小本身反映了兴趣的强度
        return (scores.unsqueeze(-1) * history).sum(dim=1)  # [B, dim]
