"""MMoE 的参考实现。手写时不要看；写完后用 `uv run drill diff mmoe --ref` 对照。"""

import torch
import torch.nn as nn


class MMoE(nn.Module):
    def __init__(self, in_dim, expert_dim, num_experts=4, num_tasks=2):
        super().__init__()
        # 专家：所有任务共享，每个就是一个普通的多层感知机
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, expert_dim), nn.ReLU(),
                          nn.Linear(expert_dim, expert_dim))
            for _ in range(num_experts)])
        # gate：每个任务一个（单任务的 MoE 只有一个 gate，这是二者的区别）
        self.gates = nn.ModuleList([nn.Linear(in_dim, num_experts) for _ in range(num_tasks)])

    def forward(self, x):  # x: [B, in_dim]
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, expert_dim]
        task_inputs = []
        for gate in self.gates:
            # 第 k 个任务的权重：softmax(W_gk x)，在专家这一维上归一化
            weight = torch.softmax(gate(x), dim=-1)  # [B, num_experts]
            # 加权求和：权重补一维后与专家输出相乘，再沿专家维求和
            task_inputs.append((weight.unsqueeze(-1) * expert_out).sum(dim=1))  # [B, expert_dim]
        return task_inputs  # 每个任务一个表示，之后各自接任务塔
