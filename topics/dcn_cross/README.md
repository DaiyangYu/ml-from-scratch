# DCN-v2：交叉层与交叉网络

建议用时：5 分钟

## 要实现什么

在 `dcn_cross.py` 中实现 DCN-v2 论文的显式特征交叉部分（矩阵形式）：

- `CrossLayerV2`：一层交叉。输入是最初的特征向量 `x0` 和上一层的输出 `xl`，输出下一层的结果。
- `CrossNetV2`：把若干层交叉串起来。第一层的“上一层输出”就是 `x0` 本身。

`x0` 是全部特征的 embedding 拼平之后的向量。

## 接口

`CrossLayerV2(dim)`

| 属性 | 定义 |
| --- | --- |
| `linear` | `nn.Linear(dim, dim)`，即这一层的权重矩阵和偏置 |

`forward(x0, xl)`：两个输入都是 `[B, dim]`，返回 `[B, dim]`。

`CrossNetV2(dim, num_layers=3)`

| 属性 | 定义 |
| --- | --- |
| `layers` | `nn.ModuleList`，包含 `num_layers` 个 `CrossLayerV2(dim)` |

`forward(x0)`：输入 `[B, dim]`，返回最后一层的输出 `[B, dim]`。

## 不要求

- DCN 第一版（每层只有一个权重向量）的写法
- 低秩分解、混合专家版本
- 与 DNN 分支的并行或串行组合

## 检验方式

- 单层：取出权重矩阵和偏置，按论文公式逐样本手算后对比。
- 多层：用逐层手算的结果对比。
- 与参考实现在相同权重下对比。
