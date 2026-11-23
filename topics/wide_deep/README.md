# Wide & Deep

建议用时：10 分钟

## 要实现什么

在 `wide_deep.py` 中实现 `WideAndDeep(nn.Module)`，输出点击率预估的 logit：

- Wide 部分：对人工交叉后的稀疏特征向量 `wide_x` 做一层线性变换；
- Deep 部分：每个样本有 `n_fields` 个特征域，每个域一个特征 id，查同一张 Embedding 表后送入多层感知机；
- 两部分的结果合成一个 logit。

注意区分两个量：`num_features` 是全部特征 id 的总数（Embedding 表的行数），`n_fields` 是每个样本的特征域个数。

## 接口

构造：`WideAndDeep(wide_dim, num_features, n_fields, emb_dim=16, hidden=64)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `wide` | `nn.Linear(wide_dim, 1)` |
| `emb` | `nn.Embedding(num_features, emb_dim)` |
| `deep` | `nn.Sequential`：一层线性变换到 `hidden` 维、`nn.ReLU()`、`nn.Linear(hidden, 1)`。第一层的输入维度需要自己确定 |

`forward(wide_x, feat_ids)`：

- `wide_x`：`[B, wide_dim]` 的浮点张量
- `feat_ids`：`[B, n_fields]` 的整数张量
- 返回 `[B, 1]` 的 logit（不过 sigmoid）

## 不要求

- 特征交叉的构造、特征列定义
- Wide 与 Deep 使用不同优化器的联合训练
- 损失函数

## 检验方式

- 取出各层权重，按结构逐样本手算后对比。
- 与参考实现在相同权重下对比，其中包含 `num_features` 与 `n_fields` 不相等的情况。
