"""MicroGrad 神经网络部分的参考实现。手写时不要看；写完后用 `uv run drill diff micrograd_nn --ref` 对照。"""

import random

from micrograd_nn_given import Value


class Module:
    def zero_grad(self):
        # 梯度在 backward 时是累加的，每次反向传播之前都要清零
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        # 权重必须随机：全部相同的话，同一层的神经元得到的梯度也相同，永远学不出差别
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # sum 的第二个参数是起始值，把偏置放在这里，结果就是 w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]  # 返回的是 Value 对象本身，不是数值的副本


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]  # 每个神经元接收同一个输入
        return out[0] if len(out) == 1 else out  # 只有一个输出时去掉列表，方便直接当作标量使用

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts  # 第 i 层的输入维度是 sizes[i]，输出维度是 sizes[i + 1]
        # 最后一层不加非线性：输出要能取任意实数，交给损失函数处理
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # 上一层的输出是下一层的输入
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
