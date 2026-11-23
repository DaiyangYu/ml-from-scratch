# AUGRU：带注意力更新门的 GRU 单元

建议用时：15 分钟

## 要实现什么

在 `augru.py` 中实现 `AUGRUCell(nn.Module)`，即 DIEN 兴趣演化层使用的单步单元。

它在普通 GRU 单元的基础上多接收一个注意力分数 `a`（目标 item 与这一步兴趣的相关程度），用 `a` 缩放更新门，再做新旧状态的融合：相关程度越低，这一步对隐状态的更新越少。

GRU 部分的公式和参数布局与 `nn.GRUCell` 一致。注意力分数的用法按 DIEN 原论文的公式。

注意：论文和 PyTorch 对更新门的记号方向相反，照搬论文的式子套到 PyTorch 的记号上会出错。

## 接口

构造：`AUGRUCell(input_dim, hidden_dim)`

参数名（测试按名字读写权重，必须一致），布局与 `nn.GRUCell` 相同：

| 属性 | 定义 |
| --- | --- |
| `weight_ih` | `nn.Parameter`，形状 `[3 * hidden_dim, input_dim]` |
| `weight_hh` | `nn.Parameter`，形状 `[3 * hidden_dim, hidden_dim]` |
| `bias_ih` | `nn.Parameter`，形状 `[3 * hidden_dim]` |
| `bias_hh` | `nn.Parameter`，形状 `[3 * hidden_dim]` |

`forward(x, h, att_score)`：`x` 形状 `[B, input_dim]`，`h` 形状 `[B, hidden_dim]`，`att_score` 形状 `[B, 1]`、取值在 0 到 1 之间，返回形状 `[B, hidden_dim]` 的新隐状态。

## 必须正确处理的情况

- `att_score` 为 1 时，结果与普通 GRU 单元（`nn.GRUCell`）相同。
- `att_score` 为 0 时，隐状态保持不变。

## 不要求

- 注意力分数本身的计算
- 兴趣抽取层、辅助损失、沿时间步展开

## 说明

这里按论文公式实现。DIEN 官方 TensorFlow 代码的写法是 `u = (1 - att_score) * u`，其中 `u` 乘在旧状态上，与论文公式并不相同，本题不采用。

## 检验方式

- 装入 `nn.GRUCell` 的权重，用官方单元的输出推算出应有的结果后对比。
- 与参考实现在相同权重下对比。
