# ESMM：全空间多任务模型

建议用时：10 分钟

## 要实现什么

在 `esmm.py` 中实现 ESMM 论文的模型输出和损失：

- `ESMM(nn.Module)`：输入是共享的特征表示，两个结构相同的塔分别给出点击率 `pCTR` 和转化率 `pCVR`，再由它们得到“曝光后点击并转化”的概率 `pCTCVR`。
- `esmm_loss`：ESMM 的训练损失，全部样本（全部曝光）都参与计算。

## 接口

构造：`ESMM(in_dim, hidden=64)`

参数名（测试按名字读写权重，必须一致）：

| 属性 | 定义 |
| --- | --- |
| `ctr_tower` | `nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())` |
| `cvr_tower` | 结构同上 |

`forward(x)`：`x` 的形状是 `[B, in_dim]`，返回三元组 `(p_ctr, p_cvr, p_ctcvr)`，每项都是形状 `[B]` 的概率。

`esmm_loss(p_ctr, p_ctcvr, click, conversion)`：

- `p_ctr`、`p_ctcvr`：`[B]` 的概率
- `click`：`[B]`，是否点击，取值 0 或 1
- `conversion`：`[B]`，是否点击并且转化，取值 0 或 1
- 两个标签可能是整数、布尔或浮点类型
- 返回一个标量。损失由若干项二元交叉熵组成，每一项都对 batch 取平均，各项直接相加、不加权

## 不要求

- 共享 Embedding 层（这里直接以共享表示 `x` 为输入）
- 训练循环、评估

## 检验方式

- 取出两个塔的权重逐样本手算概率，并核对三个输出之间的关系。
- 损失与手写的二元交叉熵公式对比。
- 与参考实现在相同权重下对比。
