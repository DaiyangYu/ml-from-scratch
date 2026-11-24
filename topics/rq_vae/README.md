# RQ-VAE：残差量化

建议用时：15 分钟

## 要实现什么

在 `rq_vae.py` 中实现 `ResidualQuantizer(nn.Module)`，即 RQ-VAE 中间的残差量化部分。TIGER 用它把 item 的向量变成一串离散的码（语义 ID）。

共有 `num_levels` 层，每层有一个独立的码本，码本里有 `codebook_size` 个 `dim` 维的码字。对输入向量 `z`：

```
r_1 = z
对第 l 层（l = 1 … num_levels）：
    c_l = 第 l 层码本中与 r_l 的欧氏距离最小的码字的编号
    e_l = 第 l 层码本的第 c_l 个码字
    r_{l+1} = r_l − e_l
```

`(c_1, …, c_L)` 就是这个向量的语义 ID，量化结果是各层码字之和 `Σ_l e_l`。

损失（`sg` 表示停止梯度，`mse` 表示对全部元素取平均的均方误差）：

```
loss = Σ_l [ mse(sg[r_l], e_l) + beta × mse(r_l, sg[e_l]) ]
```

前一项更新码本，后一项（commitment loss）约束输入。各层的损失相加，不取平均。

梯度的要求：

- 取最近码字这一步不可导。返回的 `z_q` 数值上等于 `Σ_l e_l`，但反向传播时，`z_q` 对 `z` 的梯度要等同于恒等映射（直通估计），并且 `z_q` 不向码本传梯度。
- 每一层的码本只从本层的损失项得到梯度：计算下一层的残差时，不让梯度经过 `e_l` 传回码本。

## 接口

构造：`ResidualQuantizer(num_levels, codebook_size, dim, beta=0.25)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `codebooks` | `nn.ModuleList`，含 `num_levels` 个 `nn.Embedding(codebook_size, dim)` |

`forward(z)`：`z` 是形状 `[B, dim]` 的张量，返回 `(z_q, loss, codes)`：

- `z_q`：形状 `[B, dim]`
- `loss`：标量张量
- `codes`：形状 `[B, num_levels]` 的整数张量（`torch.long`），第 `l` 列是第 `l` 层的码字编号

## 不要求

- 编码器、解码器和重构损失
- 用 k-means 初始化码本（见题目 `kmeans`）
- Sinkhorn 均衡分配、码本的指数滑动平均更新
- 语义 ID 冲突的处理

## 检验方式

- 逐样本、逐层、逐码字按定义循环计算，对比码、量化结果和损失。
- 按定义手工推出损失对 `z` 和对各层码本的梯度，与自动求导的结果对比。
- 与参考实现在相同权重下对比。
