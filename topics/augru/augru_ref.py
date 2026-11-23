"""AUGRU 的参考实现。手写时不要看；写完后用 `uv run drill diff augru --ref` 对照。"""

import torch
import torch.nn as nn


class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 与 nn.GRUCell 相同的布局：三个门按 r、z、n 的顺序摞在一起
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_dim))
        bound = hidden_dim ** -0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x, h, att_score):  # att_score: [B, 1]
        # 这一段就是 nn.GRUCell
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        # 论文的更新门 u 是「取新状态的比例」：h_t = (1 - u) * h_{t-1} + u * h~_t，
        # 而 PyTorch 的 z 是「保留旧状态的比例」，所以 u 对应 1 - z。
        # AUGRU 的改动只有一行：u' = a * u。不能写成 a * z，那样 a = 0 时旧状态会被候选状态完全覆盖。
        u = att_score * (1 - z)
        return (1 - u) * h + u * n
