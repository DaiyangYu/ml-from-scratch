"""RQ-VAE 残差量化的参考实现。手写时不要看；写完后用 `uv run drill diff rq_vae --ref` 对照。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, num_levels, codebook_size, dim, beta=0.25):
        super().__init__()
        self.beta = beta
        # 每层一个独立的码本；后面的层量化的是前面留下的残差，数值越来越小
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_levels)])

    def forward(self, z):  # [B, dim]
        residual = z
        quantized = torch.zeros_like(z)
        loss = z.new_zeros(())
        codes = []
        for codebook in self.codebooks:
            # 残差与每个码字的欧氏距离的平方：[B, 1, dim] - [1, K, dim] -> [B, K]
            distances = (residual.unsqueeze(1) - codebook.weight.unsqueeze(0)).pow(2).sum(dim=-1)
            index = distances.argmin(dim=-1)  # [B]
            e = codebook(index)               # [B, dim]
            # 两项数值相同，区别只在梯度流向：前一项把码字拉向残差，后一项（乘 beta）把残差拉向码字
            loss = loss + F.mse_loss(e, residual.detach()) + self.beta * F.mse_loss(residual, e.detach())
            quantized = quantized + e
            # 减去的是 detach 之后的码字：否则后面各层的损失会经过残差传回本层的码本
            residual = residual - e.detach()
            codes.append(index)
        # 直通估计：数值等于各层码字之和，梯度原样传给 z，且不经过码本
        z_q = z + (quantized - z).detach()
        return z_q, loss, torch.stack(codes, dim=-1)
