"""DCN-v2 交叉层的参考实现。手写时不要看；写完后用 `uv run drill diff dcn_cross --ref` 对照。"""

import torch
import torch.nn as nn


class CrossLayerV2(nn.Module):
    """x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l"""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)  # W_l 和 b_l。第一版这里只是一个权重向量，交叉强度是标量

    def forward(self, x0, xl):  # 均为 [B, dim]
        # 两个易错点：逐元素相乘的对象是最初的 x0，不是上一层输出；最后要加上残差 xl
        return x0 * self.linear(xl) + xl


class CrossNetV2(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayerV2(dim) for _ in range(num_layers)])

    def forward(self, x0):
        xl = x0  # 第一层的“上一层输出”就是 x0
        for layer in self.layers:
            xl = layer(x0, xl)  # 每一层都传入同一个 x0，每过一层交叉阶数加一
        return xl
