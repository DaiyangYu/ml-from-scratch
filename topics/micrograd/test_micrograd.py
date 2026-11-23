"""MicroGrad 的测试：把 backward() 的结果与数值微分对比。只依赖 README 中约定的公开接口。"""

import math

import pytest

from micrograd import Value

TOL = 1e-6


def numeric_grads(f, xs, h=1e-6):
    """用中心差分求 f 对每个输入的偏导数。f 接收若干 Value，返回一个 Value。"""
    grads = []
    for i in range(len(xs)):
        plus = [x + h if j == i else x for j, x in enumerate(xs)]
        minus = [x - h if j == i else x for j, x in enumerate(xs)]
        f_plus = f(*[Value(x) for x in plus]).data
        f_minus = f(*[Value(x) for x in minus]).data
        grads.append((f_plus - f_minus) / (2 * h))
    return grads


def check(f, xs):
    """对 f 在 xs 处调用 backward()，并与数值微分逐项比较。"""
    inputs = [Value(x) for x in xs]
    out = f(*inputs)
    out.backward()
    expected = numeric_grads(f, xs)
    for i, (value, grad) in enumerate(zip(inputs, expected)):
        assert value.grad == pytest.approx(grad, abs=TOL), f"第 {i} 个输入的导数不对"


def test_initial_grad_is_zero():
    assert Value(3.0).grad == 0


def test_forward_values():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == pytest.approx(-1.0)
    assert (a - b).data == pytest.approx(5.0)
    assert (a * b).data == pytest.approx(-6.0)
    assert (a / b).data == pytest.approx(-2.0 / 3.0)
    assert (-a).data == pytest.approx(-2.0)
    assert (a ** 3).data == pytest.approx(8.0)
    assert a.tanh().data == pytest.approx(math.tanh(2.0))
    assert a.relu().data == pytest.approx(2.0)
    assert b.relu().data == pytest.approx(0.0)


def test_forward_with_python_numbers():
    a = Value(4.0)
    assert (a + 1).data == pytest.approx(5.0)
    assert (1 + a).data == pytest.approx(5.0)
    assert (a - 1).data == pytest.approx(3.0)
    assert (1 - a).data == pytest.approx(-3.0)
    assert (a * 2).data == pytest.approx(8.0)
    assert (2 * a).data == pytest.approx(8.0)
    assert (a / 2).data == pytest.approx(2.0)
    assert (2 / a).data == pytest.approx(0.5)
    assert (a ** 0.5).data == pytest.approx(2.0)


def test_output_grad_is_one():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert out.grad == pytest.approx(1.0)


def test_add_and_mul():
    a, b = Value(2.0), Value(-3.0)
    out = a * b + a
    out.backward()
    assert a.grad == pytest.approx(-2.0)  # b + 1
    assert b.grad == pytest.approx(2.0)   # a


def test_same_node_used_twice_in_add():
    a = Value(3.0)
    out = a + a
    out.backward()
    assert a.grad == pytest.approx(2.0)


def test_same_node_used_twice_in_mul():
    a = Value(3.0)
    out = a * a
    out.backward()
    assert a.grad == pytest.approx(6.0)


def test_sub_div_neg():
    check(lambda a, b: (a - b) / (-a), [2.0, 5.0])


def test_python_numbers_on_both_sides():
    check(lambda a: 2 * a + a * 3 - 1 + (1 - a) + 4 / a + a / 4, [1.5])


def test_pow():
    check(lambda a: a ** 3 + a ** 0.5 + a ** -1, [2.0])


def test_tanh():
    check(lambda a, b: (a * b).tanh(), [0.7, -1.3])


def test_relu_positive_and_negative():
    check(lambda a: (a * 2).relu(), [1.2])
    check(lambda a: (a * 2).relu(), [-1.2])


def test_branches_that_merge_again():
    # b 和 c 都依赖 a，d 同时依赖 b 和 c，e 又同时依赖 d 和 b：
    # 如果节点在下游导数尚未全部传回时就向上游传播，结果会出错。
    def f(a):
        b = a * 2
        c = b + a
        d = b * c
        e = d + b
        return e * e

    check(f, [1.5])


def test_long_chain_through_shared_node():
    def f(a, b):
        x = a * b
        y = x + a
        z = y * x
        w = z + y + x
        return w.tanh() + w * a

    check(f, [0.3, -0.8])


def test_single_neuron():
    def neuron(x1, x2, w1, w2, bias):
        return (x1 * w1 + x2 * w2 + bias).tanh()

    check(neuron, [2.0, 0.0, -3.0, 1.0, 6.88])


def test_compound_expression():
    def f(a, b):
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        g = e ** 2
        h = g / 2.0
        return h + 10.0 / g

    check(f, [-4.0, 2.0])
