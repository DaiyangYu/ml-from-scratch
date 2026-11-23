# DIN Attention：局部激活单元

建议用时：10 分钟

## 要实现什么

在 `din_attention.py` 中实现 `DINAttention(nn.Module)`，即 DIN 的局部激活单元（Local Activation Unit）：根据目标 item，给用户历史行为序列中的每个 item 打一个分数，再用这些分数对历史 item 的向量加权求和，得到用户对这个目标的兴趣向量。

每个历史 item 的分数由一个小型多层感知机给出。它的输入是四项的拼接，顺序与论文实现一致：目标、历史、二者之差（目标减历史）、二者之积。

按原论文的做法，分数不做归一化，直接作为权重。

## 接口

构造：`DINAttention(dim, hidden=32)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `mlp` | `nn.Sequential(nn.Linear(4 * dim, hidden), nn.PReLU(), nn.Linear(hidden, 1))` |

`forward(target, history, mask=None)`：

- `target`：形状 `[B, dim]`，目标 item 的向量
- `history`：形状 `[B, L, dim]`，历史 item 的向量
- `mask`：形状 `[B, L]`，取值为 0 的位置是 padding，非 0 的位置是真实行为；`None` 表示全部是真实行为
- 返回：形状 `[B, dim]`

## 必须正确处理的情况

- padding 位置对结果没有任何贡献，无论该位置的向量是什么。
- 某个样本的历史全部是 padding 时，结果是零向量。

## 不要求

- 完整的 DIN 模型（Embedding 层、其余特征、最后的多层感知机）
- Dice 激活函数
- 带 softmax 归一化的变体

## 检验方式

- 逐样本、逐位置按定义计算后对比。
- 与参考实现在相同权重下对比。
