# DeepFM

建议用时：15 分钟

## 要实现什么

在 `deepfm.py` 中实现 `DeepFM(nn.Module)`。每个样本有 `n_fields` 个特征域，每个域一个特征 id（简化设定：特征值都取 1）。输出是两部分之和：

```
ŷ = ŷ_FM + ŷ_DNN
ŷ_FM = w0 + Σ_i w_i + Σ_{i<j} <v_i, v_j>
```

- FM 部分的定义与题目 `fm` 相同：`w_i` 是第 `i` 个特征的一阶权重，`v_i` 是它的 `dim` 维隐向量。二阶交叉项必须用线性复杂度的化简写法，不能写两重循环。
- DNN 部分是一个多层感知机，输入是该样本全部特征域的 embedding。

## 接口

构造：`DeepFM(num_features, n_fields, dim=16, hidden=64)`

`num_features` 是全部特征 id 的总数，`n_fields` 是每个样本的特征域个数。

参数名（测试按名字读写权重，必须一致；除此之外不能有别的参数）：

| 属性 | 定义 |
| --- | --- |
| `w0` | `nn.Parameter`，形状 `[1]`，初始为 0 |
| `linear` | `nn.Embedding(num_features, 1)`，一阶权重 |
| `emb` | `nn.Embedding(num_features, dim)` |
| `dnn` | `nn.Sequential(nn.Linear(n_fields * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))` |

`forward(feat_ids)`：`feat_ids` 是形状 `[B, n_fields]` 的整数张量，返回形状 `[B, 1]` 的 logit（不过 sigmoid）。

## 不要求

- 带数值的特征、连续特征的处理
- 损失函数、训练

## 检验方式

- FM 部分按定义用两重循环逐样本计算，DNN 部分取出权重手算，二者相加后对比。
- 与参考实现在相同权重下对比。
