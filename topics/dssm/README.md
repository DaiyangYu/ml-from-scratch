# DSSM 双塔

建议用时：10 分钟

## 要实现什么

在 `dssm.py` 中实现 `DSSM(nn.Module)`：召回用的双塔模型。用户特征和物品特征各自经过一个多层感知机，映射成同一空间里的向量；用户与物品的匹配分是这两个向量的余弦相似度。

两个塔互不依赖，所以要分别提供只算用户向量、只算物品向量的方法：物品向量可以离线算好建索引，线上只算用户向量。

## 接口

构造：`DSSM(user_dim, item_dim, out_dim=64, hidden=128)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `user_tower` | `nn.Sequential(nn.Linear(user_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))` |
| `item_tower` | `nn.Sequential(nn.Linear(item_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))` |

方法：

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `encode_user(user_feats)` | `[B, user_dim]` | `[B, out_dim]`，每一行是长度为 1 的用户向量 |
| `encode_item(item_feats)` | `[B, item_dim]` | `[B, out_dim]`，每一行是长度为 1 的物品向量 |
| `forward(user_feats, item_feats)` | `[B, user_dim]`、`[B, item_dim]` | `[B]`，第 `b` 个用户与第 `b` 个物品的余弦相似度 |

## 必须正确处理的情况

- `forward` 的结果要与 `encode_user`、`encode_item` 两个输出的逐行点积一致（线上检索用的就是这两个向量）。

## 不要求

- 特征的 Embedding 处理（输入已经是稠密特征向量）
- 损失函数、负采样、温度系数

## 检验方式

- 与 `F.cosine_similarity` 以及按塔的权重手工展开的计算结果对比。
- 与参考实现在相同权重下对比。
