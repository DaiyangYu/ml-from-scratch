# MicroGrad 的神经网络部分

建议用时：15 分钟

## 要实现什么

在 `micrograd_nn.py` 中，用现成的标量自动微分引擎 `Value` 搭出多层感知机，对应 Karpathy 的 micrograd 中的 `nn.py`。共四个类：

- `Module`：基类，提供取出全部参数和把梯度清零的接口。
- `Neuron`：一个神经元，对输入做加权求和再加偏置，可选地接 ReLU。
- `Layer`：一层，由若干个接收相同输入的神经元组成。
- `MLP`：多层感知机，由若干层串联而成。隐藏层带 ReLU，最后一层不带。

`Value` 已经在骨架中导入，支持 `+`、`*`、`-`、`/`、`**`、`.relu()`、`.tanh()`、`.backward()`，有 `.data` 和 `.grad` 两个属性。

## 接口

| 类 | 构造 | 属性 | 调用 |
| --- | --- | --- | --- |
| `Module` | 无参数 | 无 | `parameters()` 返回参数列表，基类返回空列表；`zero_grad()` 把所有参数的 `grad` 置 0 |
| `Neuron(Module)` | `Neuron(nin, nonlin=True)` | `w`：长度为 `nin` 的 `Value` 列表；`b`：一个 `Value` | `neuron(x)` 返回一个 `Value` |
| `Layer(Module)` | `Layer(nin, nout, **kwargs)`，`kwargs` 原样传给每个 `Neuron` | `neurons`：长度为 `nout` 的 `Neuron` 列表 | `layer(x)` 返回长度为 `nout` 的 `Value` 列表；`nout` 为 1 时直接返回那一个 `Value` |
| `MLP(Module)` | `MLP(nin, nouts)`，`nouts` 是各层输出维度的列表 | `layers`：`Layer` 列表 | `mlp(x)` 返回最后一层的输出 |

输入 `x` 是长度为 `nin` 的列表，元素可以是 Python 数，也可以是 `Value`。

`parameters()` 的顺序（测试按顺序读写权重）：

- `Neuron`：先是 `w` 中的各个权重，最后是 `b`；
- `Layer`：按神经元的顺序依次列出；
- `MLP`：按层的顺序依次列出。

初始化：权重在 `[-1, 1]` 上均匀随机，偏置为 0。

## 必须正确处理的情况

- `parameters()` 返回的必须是模型实际使用的那些 `Value` 对象本身：修改它们的 `data`，模型的输出要随之改变。

## 不要求

- 引擎 `Value` 本身（见题目 micrograd）
- 损失函数、优化器、训练循环（测试会自己写几步梯度下降来检查梯度可用）
- ReLU 以外的激活函数
- 宽度为 1 的隐藏层（按上面的约定，这样的层返回单个 `Value` 而不是列表，下一层无法遍历；原版同样不支持）

## 检验方式

- 把同样的权重放进用 PyTorch 张量手写的多层感知机，对比输出和每个参数的梯度。
- 用几步梯度下降检查损失确实下降。
- 与参考实现在相同权重下对比。
