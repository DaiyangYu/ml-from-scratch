# GRU Cell：论文写法与 PyTorch 写法

建议用时：25 分钟

## 要实现什么

在 `gru_cell.py` 中实现两个类。二者都是单步的 GRU 单元：输入当前步的 `x` 和旧隐状态 `h`，输出新隐状态。

- `GRUCellPaper`：原论文（Cho et al., 2014）的写法。
- `GRUCellTorch`：PyTorch `nn.GRUCell` 的写法。公式和参数布局都与它一致，装入它的权重后输出相同。

两种写法只在候选状态的计算上不同，所以即使权重相同，数值也对不上；两个门的含义、新旧状态的融合方式是一样的。

## 接口

两个类的 `forward(x, h)` 相同：`x` 形状 `[B, input_dim]`，`h` 形状 `[B, hidden_dim]`，返回形状 `[B, hidden_dim]` 的新隐状态。

### `GRUCellPaper(input_dim, hidden_dim)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `Wz` | `nn.Linear(input_dim + hidden_dim, hidden_dim)`，更新门 |
| `Wr` | `nn.Linear(input_dim + hidden_dim, hidden_dim)`，重置门 |
| `Wh` | `nn.Linear(input_dim + hidden_dim, hidden_dim)`，候选状态 |

三个 Linear 的输入都是两部分的拼接，来自输入的部分在前，来自隐状态的部分在后。更新门 `z` 表示保留旧状态的比例。

### `GRUCellTorch(input_dim, hidden_dim)`

| 属性 | 定义 |
| --- | --- |
| `weight_ih` | `nn.Parameter`，形状 `[3 * hidden_dim, input_dim]` |
| `weight_hh` | `nn.Parameter`，形状 `[3 * hidden_dim, hidden_dim]` |
| `bias_ih` | `nn.Parameter`，形状 `[3 * hidden_dim]` |
| `bias_hh` | `nn.Parameter`，形状 `[3 * hidden_dim]` |

三个门在这四个参数中的排列顺序与 `nn.GRUCell` 相同。初始化方式不限。

## 不要求

- 沿时间步展开整个序列、多层、双向
- 特定的初始化方式，从 `nn.GRU` 复制权重的方法

## 检验方式

- `GRUCellPaper`：把三个 Linear 的权重拆成作用于输入和作用于隐状态的两块，按论文公式手算后对比。
- `GRUCellTorch`：装入 `nn.GRUCell` 的权重后与官方输出对比；再沿时间步展开，与 `nn.GRU` 的输出对比。
- 两个类都与参考实现在相同权重下对比。
