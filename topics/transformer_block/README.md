# Transformer Block：注意力、多头注意力、编码器块

建议用时：30 分钟

## 要实现什么

在 `transformer_block.py` 中实现三样东西，后者依次用到前者：

1. 函数 `attention`：缩放点积注意力。
2. 类 `MultiHead`：多头自注意力，查询、键、值都来自同一个输入。
3. 类 `EncoderBlock`：一个编码器块，由多头自注意力和前馈网络两个子层组成，每个子层带残差连接，采用 Pre-LN 结构。

不使用 dropout。

## 接口

### `attention(q, k, v, mask=None)`

`q` 形状 `[B, heads, Lq, d_k]`，`k`、`v` 形状 `[B, heads, Lk, d_k]`，返回形状 `[B, heads, Lq, d_k]`。

### `MultiHead(d_model=64, num_heads=4)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `Wq` | `nn.Linear(d_model, d_model)`，查询的投影 |
| `Wk` | `nn.Linear(d_model, d_model)`，键的投影 |
| `Wv` | `nn.Linear(d_model, d_model)`，值的投影 |
| `Wo` | `nn.Linear(d_model, d_model)`，各头拼接后的输出投影 |

`forward(x, mask=None)`：`x` 形状 `[B, L, d_model]`，返回同样的形状。投影后的 `d_model` 维按顺序等分给各个头，第一个头取最前面的 `d_model / num_heads` 维。

### `EncoderBlock(d_model=64, num_heads=4, d_ff=256)`

| 属性 | 定义 |
| --- | --- |
| `attn` | `MultiHead(d_model, num_heads)` |
| `ln1` | `nn.LayerNorm(d_model)`，用于注意力子层 |
| `ln2` | `nn.LayerNorm(d_model)`，用于前馈子层 |
| `ffn` | `nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))` |

`forward(x, mask=None)`：`x` 形状 `[B, L, d_model]`，返回同样的形状。

### mask 的约定

三处的 `mask` 含义相同：取值为 0 的位置表示该查询不能看到该键，非 0 表示可以看到；形状可以广播到 `[B, heads, Lq, Lk]`，例如因果 mask 的 `[1, 1, L, L]`，或屏蔽若干键的 `[B, 1, 1, L]`。`mask=None` 表示全部可见。

## 不要求

- dropout、位置编码
- 解码器、交叉注意力
- 某个查询的所有键都被屏蔽的情况

## 检验方式

- `attention` 与 `F.scaled_dot_product_attention` 对比。
- `MultiHead` 把权重搬进 `nn.MultiheadAttention` 后对比。
- `EncoderBlock` 把权重搬进 `nn.TransformerEncoderLayer(norm_first=True)` 后对比。
- 以上都覆盖无 mask、因果 mask、屏蔽部分键三种情况；另外与参考实现在相同权重下对比。
