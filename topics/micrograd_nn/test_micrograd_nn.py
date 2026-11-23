"""MicroGrad 神经网络部分的测试。独立检验的办法：把同样的权重放进用 PyTorch 张量手写的多层感知机，对比输出和梯度。"""

import random

import pytest
import torch

from micrograd_nn import MLP, Layer, Module, Neuron
from micrograd_nn_given import Value
from micrograd_nn_ref import MLP as ReferenceMLP

TOL = 1e-6


def set_parameters(model, seed):
    """按 parameters() 的顺序写入一组固定的随机数（偏置也不为 0），返回这组数。"""
    generator = random.Random(seed)
    values = [generator.uniform(-1, 1) for _ in model.parameters()]
    for parameter, value in zip(model.parameters(), values):
        parameter.data = value
    return values


def torch_mlp(values, nin, nouts, x):
    """用 PyTorch 张量按同样的参数顺序搭一个多层感知机，返回（输出列表、各参数的张量）。"""
    tensors = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in values]
    cursor = 0
    h = [torch.tensor(float(v), dtype=torch.float64) for v in x]
    sizes = [nin] + nouts
    for layer in range(len(nouts)):
        out = []
        for _ in range(sizes[layer + 1]):
            w = tensors[cursor:cursor + sizes[layer]]
            b = tensors[cursor + sizes[layer]]
            cursor += sizes[layer] + 1
            act = sum(wi * hi for wi, hi in zip(w, h)) + b
            out.append(torch.relu(act) if layer != len(nouts) - 1 else act)
        h = out
    assert cursor == len(tensors)
    return h, tensors


# ---------- Module ----------

def test_module_base_class():
    module = Module()
    assert module.parameters() == [], "基类的 parameters() 应返回空列表"
    module.zero_grad()  # 没有参数时也不应报错
    for cls in (Neuron, Layer, MLP):
        assert issubclass(cls, Module), f"{cls.__name__} 应继承 Module"


# ---------- Neuron ----------

def test_neuron_parameters():
    neuron = Neuron(4)
    assert len(neuron.w) == 4 and all(isinstance(w, Value) for w in neuron.w), "w 应是长度为 nin 的 Value 列表"
    assert isinstance(neuron.b, Value), "b 应是一个 Value"
    parameters = neuron.parameters()
    assert len(parameters) == 5, "Neuron(4) 应有 5 个参数"
    assert all(a is b for a, b in zip(parameters, neuron.w + [neuron.b])), "参数顺序应为先权重后偏置，且是同一批对象"


def test_neuron_initialization():
    random.seed(0)
    neuron = Neuron(50)
    weights = [w.data for w in neuron.w]
    assert neuron.b.data == 0, "偏置应初始化为 0"
    assert all(-1 <= w <= 1 for w in weights), "权重应在 [-1, 1] 内"
    assert len(set(weights)) > 40, "权重应是随机的，不能全部相同"
    assert min(weights) < -0.3 and max(weights) > 0.3, "权重应在 [-1, 1] 上均匀分布，正负都有"


@pytest.mark.parametrize("nonlin", [True, False])
def test_neuron_forward(nonlin):
    neuron = Neuron(3, nonlin=nonlin)
    for w, value in zip(neuron.w, [0.5, -2.0, 1.5]):
        w.data = value
    neuron.b.data = 0.25
    # 0.5*1 − 2*2 + 1.5*1 + 0.25 = −1.75；换一组输入得到 0.5*2 − 2*(−1) + 1.5*1 + 0.25 = 4.75
    negative, positive = neuron([1.0, 2.0, 1.0]), neuron([2.0, -1.0, 1.0])
    assert isinstance(positive, Value), "神经元的输出应是 Value"
    assert positive.data == pytest.approx(4.75), "加权和为正时的输出不对（是否漏了偏置？）"
    assert negative.data == pytest.approx(0.0 if nonlin else -1.75), "加权和为负时的输出不对（nonlin 是否生效？）"


def test_neuron_default_is_nonlinear():
    neuron = Neuron(1)
    neuron.w[0].data, neuron.b.data = 1.0, 0.0
    assert neuron([-3.0]).data == 0, "nonlin 默认应为 True，负的加权和经过 ReLU 应为 0"


def test_neuron_accepts_values_as_input():
    neuron = Neuron(2, nonlin=False)
    neuron.w[0].data, neuron.w[1].data, neuron.b.data = 2.0, 3.0, 1.0
    x = [Value(1.0), Value(-1.0)]
    out = neuron(x)
    out.backward()
    assert out.data == pytest.approx(0.0)
    assert x[0].grad == pytest.approx(2.0) and x[1].grad == pytest.approx(3.0), "梯度应能传回作为输入的 Value"


# ---------- Layer ----------

def test_layer_structure_and_output():
    layer = Layer(3, 4)
    assert len(layer.neurons) == 4 and all(isinstance(n, Neuron) for n in layer.neurons)
    assert all(len(n.w) == 3 for n in layer.neurons), "每个神经元的输入维度应为 nin"
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 4, "nout 大于 1 时应返回长度为 nout 的列表"
    assert len(layer.parameters()) == 4 * (3 + 1)
    expected = [p for n in layer.neurons for p in n.parameters()]
    assert all(a is b for a, b in zip(layer.parameters(), expected)), "参数应按神经元的顺序依次列出"


def test_layer_with_single_output_returns_value():
    out = Layer(3, 1)([1.0, 2.0, 3.0])
    assert isinstance(out, Value), "nout 为 1 时应直接返回 Value，而不是只有一个元素的列表"


def test_layer_passes_kwargs_to_neurons():
    layer = Layer(2, 3, nonlin=False)
    for n in layer.neurons:
        n.w[0].data, n.w[1].data, n.b.data = 1.0, 1.0, 0.0
    assert all(v.data == pytest.approx(-5.0) for v in layer([-2.0, -3.0])), "nonlin=False 应传给每个神经元"


def test_layer_neurons_are_independent():
    random.seed(1)
    layer = Layer(5, 6)
    first_weights = {n.w[0].data for n in layer.neurons}
    assert len(first_weights) == 6, "各个神经元应各自独立初始化"
    assert len({id(n) for n in layer.neurons}) == 6, "各个神经元应是不同的对象"


# ---------- MLP ----------

def test_mlp_structure():
    mlp = MLP(3, [4, 5, 2])
    assert [len(layer.neurons) for layer in mlp.layers] == [4, 5, 2], "各层的神经元个数应等于 nouts"
    assert [len(layer.neurons[0].w) for layer in mlp.layers] == [3, 4, 5], "每层的输入维度应等于上一层的输出维度"
    assert len(mlp.parameters()) == (3 * 4 + 4) + (4 * 5 + 5) + (5 * 2 + 2)
    expected = [p for layer in mlp.layers for p in layer.parameters()]
    assert all(a is b for a, b in zip(mlp.parameters(), expected)), "参数应按层的顺序依次列出"


@pytest.mark.parametrize("nin, nouts, seed", [(3, [4, 4, 1], 0), (2, [5, 3], 1), (4, [1], 2), (1, [3, 3, 3, 2], 3)])
def test_mlp_output_and_gradients_match_torch(nin, nouts, seed):
    random.seed(seed)
    mlp = MLP(nin, nouts)
    values = set_parameters(mlp, seed)
    x = [random.Random(seed + 100 + i).uniform(-2, 2) for i in range(nin)]

    out = mlp(x)
    expected_out, tensors = torch_mlp(values, nin, nouts, x)
    outputs = out if isinstance(out, list) else [out]
    assert len(outputs) == len(expected_out), "输出的个数应等于最后一层的维度"
    for mine, theirs in zip(outputs, expected_out):
        assert mine.data == pytest.approx(theirs.item(), abs=TOL), "前向输出与 PyTorch 不一致"

    # 用各输出的平方和作为标量损失，对比每个参数的梯度
    mlp.zero_grad()
    loss = sum((o * o for o in outputs), Value(0.0))
    loss.backward()
    sum(o * o for o in expected_out).backward()
    for index, (parameter, tensor) in enumerate(zip(mlp.parameters(), tensors)):
        expected_grad = 0.0 if tensor.grad is None else tensor.grad.item()
        assert parameter.grad == pytest.approx(expected_grad, abs=TOL), f"第 {index} 个参数的梯度与 PyTorch 不一致"


def test_last_layer_is_linear_and_hidden_layers_are_not():
    mlp = MLP(2, [3, 1])
    # 让隐藏层输出为正、最后一层权重为负：线性的最后一层应给出负数
    for n in mlp.layers[0].neurons:
        n.w[0].data, n.w[1].data, n.b.data = 1.0, 1.0, 0.0
    last = mlp.layers[1].neurons[0]
    for w in last.w:
        w.data = -1.0
    last.b.data = 0.0
    assert mlp([1.0, 1.0]).data == pytest.approx(-6.0), "最后一层不应加 ReLU"
    # 输入为负时隐藏层应被 ReLU 截成 0，最终输出为 0
    assert mlp([-1.0, -1.0]).data == pytest.approx(0.0), "隐藏层应加 ReLU"


def test_zero_grad():
    random.seed(0)
    mlp = MLP(3, [4, 1])
    set_parameters(mlp, 5)
    mlp([1.0, -2.0, 0.5]).backward()
    assert any(p.grad != 0 for p in mlp.parameters()), "反向传播之后应有非零梯度"
    mlp.zero_grad()
    assert all(p.grad == 0 for p in mlp.parameters()), "zero_grad 之后所有参数的梯度应为 0"


def test_parameters_are_the_live_objects():
    random.seed(0)
    mlp = MLP(2, [3, 1])
    set_parameters(mlp, 7)
    x = [0.7, -0.4]
    before = mlp(x).data
    for p in mlp.parameters():
        p.data += 0.5
    assert mlp(x).data != pytest.approx(before), "修改 parameters() 返回的对象后，模型输出应随之改变"


def test_gradient_descent_reduces_loss():
    mlp = MLP(2, [8, 1])
    set_parameters(mlp, 42)  # 用固定的权重，结果不依赖于各人实现中取随机数的顺序
    data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 2.0)]  # 目标：两个输入之和

    def loss_value():
        return sum(((mlp(x) - y) ** 2 for x, y in data), Value(0.0))

    first = loss_value().data
    for _ in range(60):
        mlp.zero_grad()
        loss = loss_value()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.02 * p.grad
    assert loss_value().data < 0.2 * first, "做了 60 步梯度下降，损失应明显下降"


# ---------- 与参考实现对比 ----------

@pytest.mark.parametrize("nin, nouts, seed", [(3, [4, 4, 1], 0), (5, [2, 6, 3], 1), (2, [3, 1], 2)])
def test_matches_reference(nin, nouts, seed):
    random.seed(seed)
    mlp, reference = MLP(nin, nouts), ReferenceMLP(nin, nouts)
    assert len(mlp.parameters()) == len(reference.parameters()), "参数个数与参考实现不一致"
    set_parameters(mlp, seed)
    set_parameters(reference, seed)
    for trial in range(3):
        x = [random.Random(seed * 10 + trial + i).uniform(-2, 2) for i in range(nin)]
        mine, theirs = mlp(x), reference(x)
        mine = mine if isinstance(mine, list) else [mine]
        theirs = theirs if isinstance(theirs, list) else [theirs]
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert a.data == pytest.approx(b.data, abs=TOL), "输出与参考实现不一致"
