"""MMoE：多门控混合专家。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class MMoE(nn.Module):
    def __init__(self, in_dim, expert_dim, num_experts=4, num_tasks=2):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):  # [B, in_dim] -> 长度为 num_tasks 的列表，每项 [B, expert_dim]
        raise NotImplementedError
