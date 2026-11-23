"""SASRec 的参考实现。手写时不要看；写完后用 `uv run drill diff sasrec --ref` 对照。"""

import torch
import torch.nn as nn

from sasrec_given import EncoderBlock


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)  # 0 留给 padding
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([EncoderBlock(dim, num_heads, 4 * dim) for _ in range(num_layers)])
        # Pre-LN 的 Block 输出没有经过归一化，所以堆叠之后要再补一次 LayerNorm
        self.ln = nn.LayerNorm(dim)

    def forward(self, seq):  # seq: [B, L]，左侧 padding，0 是 pad
        B, L = seq.shape
        keep = (seq != 0).unsqueeze(-1)  # [B, L, 1]，真实 item 为 True
        pos = torch.arange(L, device=seq.device).unsqueeze(0)  # [1, L]
        x = self.item_emb(seq) + self.pos_emb(pos)  # [B, L, dim]
        # padding 位置加上位置向量后不再是 0，要重新置零
        x = x * keep  # 进入第一个 Block 之前
        # 因果 mask：下三角为 1，位置 t 只能看到不晚于 t 的位置；形状 [1, 1, L, L]，可广播到各个样本和各个头。
        # padding 不能乘进这个 mask：左侧 padding 的那些查询会一个键都看不到，softmax 得到 NaN，并且会传染给后面的层
        causal = torch.tril(torch.ones(L, L, device=seq.device)).view(1, 1, L, L)
        for block in self.blocks:
            x = block(x, causal)
            x = x * keep
        return self.ln(x)  # 位置 t 的输出用来预测第 t + 1 个 item

    def score(self, seq, cand_items):  # cand_items: [B, K]
        # 左侧 padding 保证了最后一个位置一定是最近的一次真实行为
        seq_out = self.forward(seq)[:, -1]  # [B, dim]
        cand_emb = self.item_emb(cand_items)  # [B, K, dim]，候选 item 与序列里的 item 共用一张表
        return (seq_out.unsqueeze(1) * cand_emb).sum(-1)  # [B, K]
