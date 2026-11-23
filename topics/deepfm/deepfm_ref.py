"""DeepFM 的参考实现。手写时不要看；写完后用 `uv run drill diff deepfm --ref` 对照。"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_features, n_fields, dim=16, hidden=64):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(num_features, 1)  # 一阶权重 w_i
        # FM 分支和 DNN 分支共用这一张表，这是 DeepFM 相对 Wide & Deep 的改进点；不要给两个分支各建一张
        self.emb = nn.Embedding(num_features, dim)
        self.dnn = nn.Sequential(
            nn.Linear(n_fields * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, feat_ids):  # [B, n_fields]
        v = self.emb(feat_ids)                               # [B, n_fields, dim]，两个分支都从它出发
        # FM 分支：一阶项 + 二阶交叉项（低阶交叉）
        linear_part = self.linear(feat_ids).sum(dim=1)       # [B, 1]
        square_of_sum = v.sum(dim=1).pow(2)                  # [B, dim] 先求和再平方
        sum_of_square = v.pow(2).sum(dim=1)                  # [B, dim] 先平方再求和
        cross = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)  # [B, 1]
        # DNN 分支：同一个 v 拼平后送入多层感知机（高阶交叉）
        dnn_out = self.dnn(v.flatten(1))                     # [B, 1]
        return self.w0 + linear_part + cross + dnn_out
