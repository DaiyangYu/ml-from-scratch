# MMoE：多门控混合专家

建议用时：10 分钟

## 要实现什么

在 `mmoe.py` 中实现 `MMoE(nn.Module)`，即 MMoE 论文中任务塔之前的部分：

- 一组所有任务共享的专家，每个专家是一个两层的多层感知机；
- 每个任务有一个自己的 gate，由输入算出对各个专家的权重；
- 每个任务得到的表示，是各专家输出按该任务的权重加权之后的结果。

## 接口

构造：`MMoE(in_dim, expert_dim, num_experts=4, num_tasks=2)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `experts` | `nn.ModuleList`，包含 `num_experts` 个专家，每个是 `nn.Sequential(nn.Linear(in_dim, expert_dim), nn.ReLU(), nn.Linear(expert_dim, expert_dim))` |
| `gates` | `nn.ModuleList`，包含 `num_tasks` 个 `nn.Linear(in_dim, num_experts)` |

`forward(x)`：`x` 的形状是 `[B, in_dim]`，返回一个长度为 `num_tasks` 的列表，第 `k` 项是第 `k` 个任务的表示，形状 `[B, expert_dim]`。

## 必须正确处理的情况

- `expert_dim` 与 `num_experts` 相等或不相等都要正确。
- 只有一个任务、只有一个专家的情况。

## 不要求

- 各任务的任务塔、损失函数、训练
- PLE 等后续结构

## 检验方式

- 取出专家和 gate 的权重，按论文公式逐样本手算后对比。
- 与参考实现在相同权重下对比。
