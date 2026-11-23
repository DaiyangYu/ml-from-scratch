"""MicroGrad 的神经网络部分。要求见同目录下的 README.md。"""

import random

from micrograd_nn_given import Value  # 现成的自动微分引擎，不需要重写


class Module:
    def zero_grad(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class MLP(Module):
    def __init__(self, nin, nouts):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError
