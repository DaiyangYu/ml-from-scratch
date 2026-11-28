"""FM：因子分解机。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_features, dim=16):
        super().__init__()
        self.linear = nn.Embedding(num_features,1)
        self.emb = nn.Embedding(num_features,dim)
        self.w0 = nn.Parameter(torch.zeros(1))

    def forward(self, feat_ids):  # [B, n] -> [B, 1]
        linear_part = self.linear(feat_ids).sum(dim=1)  # [B,1]
        v = self.emb(feat_ids)  # [B,n,dim]
        square_of_sum = v.sum(dim=1).pow(2) # [B,dim]
        sum_of_square = v.pow(2).sum(dim=1) # [B,dim]
        cross = 0.5*(square_of_sum-sum_of_square).sum(dim=1,keepdim=True)  # [B,1]
        return self.w0+linear_part+cross


