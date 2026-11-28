"""DeepFM。要求见同目录下的 README.md。"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_features, n_fields, dim=16, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(num_features,dim)
        self.linear = nn.Embedding(num_features,1)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.dnn = nn.Sequential(
            nn.Linear(n_fields*dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )

    def forward(self, feat_ids):  # [B, n_fields] -> [B, 1]
        linear_part = self.linear(feat_ids).sum(dim=1)  # [B, n_fields, 1]
        v = self.emb(feat_ids) # [B, n_fields, dim]
        square_of_sum = v.sum(dim=1).pow(2)  # [B, dim]
        sum_of_square = v.pow(2).sum(dim=1)  # [B, dim]
        cross = 0.5*(square_of_sum-sum_of_square).sum(dim=1,keepdim=True)
        dnn_out = self.dnn(v.flatten(1))
        return self.w0+linear_part+cross+dnn_out


