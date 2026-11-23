# FM：因子分解机

建议用时：15 分钟

## 要实现什么

在 `fm.py` 中实现 `FM(nn.Module)`。每个样本有 `n` 个特征 id（简化设定：特征值都取 1），输出

```
ŷ = w0 + Σ_i w_i + Σ_{i<j} <v_i, v_j>
```

其中 `w_i` 是第 `i` 个特征的一阶权重，`v_i` 是它的 `dim` 维隐向量。

二阶交叉项必须用线性复杂度 `O(n · dim)` 的化简写法，不能写两重循环。

## 接口

构造：`FM(num_features, dim=16)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `w0` | `nn.Parameter`，形状 `[1]`，初始为 0 |
| `linear` | `nn.Embedding(num_features, 1)`，一阶权重 |
| `emb` | `nn.Embedding(num_features, dim)`，隐向量 |

`forward(feat_ids)`：`feat_ids` 是形状 `[B, n]` 的整数张量，返回形状 `[B, 1]` 的张量。

## 不要求

- 带数值的特征（`x_i` 不为 1 的情况）
- 输出过 sigmoid、损失函数、训练

## 检验方式

- 与按定义写的两重循环逐样本对比。
- 与参考实现在相同权重下对比。
