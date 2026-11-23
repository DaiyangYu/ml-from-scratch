# 批内负采样 Softmax 损失

建议用时：10 分钟

## 要实现什么

在 `inbatch_softmax.py` 中实现函数 `inbatch_softmax_loss`：双塔召回模型最常用的训练损失。

一个 batch 里有 `B` 个（用户，正样本物品）对。对第 `i` 个用户来说，第 `i` 个物品是正样本，同一批里其余 `B - 1` 个物品都当作负样本。于是每个用户面对一个 `B` 选 1 的分类问题，损失就是这个分类问题的交叉熵，最后对 batch 取平均。

- 用户与物品之间的打分是两个向量的内积除以温度 `temperature`。
- 函数内部不做向量归一化（要不要归一化由调用方决定）。

**采样偏差纠正（logQ 纠偏）**

批内负样本不是均匀采样得到的：一个物品越热门，出现在 batch 里的机会越多，被当作负样本的次数也越多，模型会因此过度打压热门物品。纠正办法出自 Yi 等人 2019 年的论文 *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*：

- `item_log_probs` 给出 batch 中每个物品被采样到的概率的对数 `log p`；
- 每个物品的打分（已经除过温度）都要减去它自己的 `log p`，然后再算交叉熵；
- 这个修正对正样本那一项同样适用，不是只修正负样本。

`item_log_probs` 为 `None` 时不做纠偏。

## 接口

`inbatch_softmax_loss(user_vecs, item_vecs, temperature=0.05, item_log_probs=None)`

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `user_vecs` | `[B, d]` | 用户塔的输出 |
| `item_vecs` | `[B, d]` | 物品塔的输出，第 `i` 行是第 `i` 个用户的正样本 |
| `temperature` | 标量 | 温度 |
| `item_log_probs` | `[B]` 或 `None` | 各物品采样概率的对数 |

返回标量张量（0 维），梯度能够传回 `user_vecs` 和 `item_vecs`。

## 不要求

- 采样概率的估计方法
- 去除批内的假负样本（同一物品在一个 batch 中出现多次）
- 用户到物品、物品到用户的双向损失

## 检验方式

- 与手算值、逐个用户手写 log-softmax 的结果对比。
- 检查纠偏的方向：负样本越热门，它带来的惩罚越小。
- 与参考实现对比。
