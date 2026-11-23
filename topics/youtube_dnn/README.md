# YouTubeDNN 召回

建议用时：10 分钟

## 要实现什么

在 `youtube_dnn.py` 中实现 `YouTubeDNN(nn.Module)`：把“预测用户下一个观看的视频”当作多分类问题的召回模型，这里只写打分部分。

- 用户侧：把用户观看历史中各个物品的向量取平均（只对真实的历史求平均，padding 位置不参与），再经过一个多层感知机得到用户向量。
- 物品侧：没有塔，物品向量就是 Embedding 表里的那一行；历史物品和候选物品共用同一张表。
- 打分：用户向量与每个候选物品向量的点积，作为该候选的 logit。

## 接口

构造：`YouTubeDNN(num_items, dim=64, hidden=128)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `item_emb` | `nn.Embedding(num_items, dim)` |
| `user_tower` | `nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))` |

`forward(hist_ids, hist_mask, cand_ids)`：

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `hist_ids` | `[B, L]` | 历史物品 id，长度不足的用任意 id 补齐 |
| `hist_mask` | `[B, L]` | 真实历史为 1（或 `True`），padding 为 0（或 `False`）；可能是布尔、整数或浮点类型 |
| `cand_ids` | `[B, K]` | 候选物品 id |

返回形状 `[B, K]` 的 logits。

## 必须正确处理的情况

- padding 位置上的 id 是什么都不能影响结果。
- 某个用户的历史全部是 padding 时，结果不能出现 `nan`：此时历史的平均向量按全零处理。

## 不要求

- sampled softmax 损失、负采样（训练时对这 `K` 个 logits 做交叉熵即可，不在本题范围内）
- 用户画像等其他特征
- 线上的近邻检索

## 检验方式

- 与按定义逐样本计算的结果对比。
- 与参考实现在相同权重下对比。
