# Skip-gram + 负采样

建议用时：10 分钟

## 要实现什么

在 `skipgram.py` 中实现 `SkipGramNS(nn.Module)`：Word2Vec 的 Skip-gram 模型，用负采样目标函数训练。`forward` 直接返回损失。

模型有两张词向量表：词作为中心词时用一张，作为上下文词时用另一张。一对词的得分是中心词向量与上下文词向量的点积。负采样目标函数希望真实出现的（中心词，上下文词）对得分高，随机采样的（中心词，负样本词）对得分低。

损失的归约方式（按原论文）：

- 每个样本的损失 = 正样本对那一项 + 该样本 `K` 个负样本各自那一项之**和**（对 `K` 求和，不取平均）；
- 最终损失 = 各样本损失对 batch 取平均。

## 接口

构造：`SkipGramNS(vocab_size, dim=100)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `center_emb` | `nn.Embedding(vocab_size, dim)`，中心词向量表 |
| `context_emb` | `nn.Embedding(vocab_size, dim)`，上下文词向量表 |

`forward(center, pos_context, neg_contexts)`：

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `center` | `[B]` | 中心词 id |
| `pos_context` | `[B]` | 真实的上下文词 id |
| `neg_contexts` | `[B, K]` | 每个样本采到的 `K` 个负样本词 id |

返回标量张量（0 维）。

## 不要求

- 负样本的采样过程、高频词下采样
- 从语料中构造训练样本

## 检验方式

- 与手算值、按原论文目标函数逐样本逐项计算的结果对比。
- 与参考实现在相同权重下对比。
