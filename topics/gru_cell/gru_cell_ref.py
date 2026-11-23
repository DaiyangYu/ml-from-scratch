"""GRU Cell 的参考实现。手写时不要看；写完后用 `uv run drill diff gru_cell --ref` 对照。"""

import torch
import torch.nn as nn


class GRUCellPaper(nn.Module):
    """原论文写法：重置门先乘旧状态，再做线性变换。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):  # x: [B, input_dim], h: [B, hidden_dim]
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.Wz(xh))  # 更新门：保留旧状态的比例
        r = torch.sigmoid(self.Wr(xh))  # 重置门
        # 与 PyTorch 写法唯一的不同：r 直接乘在旧状态 h 上，然后才进 Linear
        h_tilde = torch.tanh(self.Wh(torch.cat([x, r * h], dim=-1)))
        # z 乘旧状态，1 - z 乘候选状态
        return z * h + (1 - z) * h_tilde


class GRUCellTorch(nn.Module):
    """nn.GRUCell 的写法：重置门乘在「旧状态的线性变换加偏置」之后。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 三个门的参数按 r、z、n 的顺序沿第 0 维摞在一起，不是 z、r、n
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_dim))
        bound = hidden_dim ** -0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x, h):  # x: [B, input_dim], h: [B, hidden_dim]
        gi = x @ self.weight_ih.T + self.bias_ih  # [B, 3 * hidden_dim]
        gh = h @ self.weight_hh.T + self.bias_hh  # [B, 3 * hidden_dim]
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        # r 乘的是 h_n（已含偏置 b_hn），不是 h 本身；输入侧的偏置 b_in 在乘法之外
        n = torch.tanh(i_n + r * h_n)
        # z 同样是保留旧状态的比例，与论文写法的融合式相同
        return (1 - z) * n + z * h
