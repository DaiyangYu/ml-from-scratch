# SASRec：自注意力序列推荐

建议用时：20 分钟

## 要实现什么

在 `sasrec.py` 中实现 `SASRec(nn.Module)`：把用户的行为序列编码成每个位置上的表示，位置 `t` 的表示只依赖位置 `t` 及之前的 item，用来预测下一个 item。

结构：item 向量加上可学习的位置向量，依次经过若干个 Transformer Block，最后过一次 LayerNorm。

Transformer Block 不需要重写，`sasrec_given.py` 已经提供，骨架里已经导入：

- `EncoderBlock(d_model, num_heads, d_ff)`：Pre-LN 的编码器块，不含 dropout。
- 调用方式 `block(x, mask)`：`x` 形状 `[B, L, d_model]`；`mask` 中取值为 0 的位置表示该查询不能看到该键，形状可以广播到 `[B, heads, L, L]`；`mask=None` 表示全部可见。

## 接口

构造：`SASRec(num_items, max_len=50, dim=64, num_heads=4, num_layers=2)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `item_emb` | `nn.Embedding(num_items + 1, dim, padding_idx=0)`，item 的 id 从 1 开始，0 留给 padding |
| `pos_emb` | `nn.Embedding(max_len, dim)`，序列的第 `t` 个位置（从 0 开始）用第 `t` 行 |
| `blocks` | `nn.ModuleList`，包含 `num_layers` 个 `EncoderBlock(dim, num_heads, 4 * dim)` |
| `ln` | `nn.LayerNorm(dim)`，作用在最后一个 Block 的输出上 |

`forward(seq)`：`seq` 是形状 `[B, L]` 的整数张量，`L` 不超过 `max_len`，序列在左侧用 0 补齐。返回形状 `[B, L, dim]`，即经过最后的 `ln` 之后各位置的表示。

`score(seq, cand_items)`：`cand_items` 是形状 `[B, K]` 的候选 item id。用序列最后一个位置的表示与每个候选 item 的向量做点积，返回形状 `[B, K]` 的分数。候选 item 的向量取自 `item_emb`。

## padding 的处理

按官方实现的做法，padding 位置的隐状态在以下时刻置零：

- item 向量与位置向量相加之后、进入第一个 Block 之前；
- 每个 Block 的输出上。

最后的 `ln` 之后不再置零。padding 不通过注意力的 mask 处理。

## 必须正确处理的情况

- 位置 `t` 的输出不受位置 `t` 之后的 item 影响。
- 序列左侧有 padding 时，输出中不能出现 NaN。

## 不要求

- 重写 Transformer Block
- dropout、对 item 向量的缩放
- 损失函数、负采样、训练

## 检验方式

- 修改序列后面的 item，检查前面位置的输出不变。
- 检查 padding 位置的输出，以及没有 Block 时的输出，是否等于按规定手算的结果。
- `score` 与「最后位置的表示点乘候选 item 向量」对比。
- 与参考实现在相同权重下对比。
