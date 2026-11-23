"""MicroGrad 的参考实现。手写时不要看；写完后用 `uv run drill diff micrograd --ref` 对照。"""

import math


class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None  # 把自身的 grad 分配给各个输入的函数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # 拓扑排序：保证处理一个节点时，它的所有下游节点都已经把导数传回来了
        order, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    # 其余运算都用加法、乘法、幂组合出来
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1
