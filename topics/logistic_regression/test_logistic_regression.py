"""逻辑回归的测试。"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from logistic_regression import LogisticRegression
from logistic_regression_ref import LogisticRegression as Reference


def random_data(seed, n=40, d=3, separable=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d) * 2
    logits = X @ true_w + 0.7
    if separable:
        y = (logits > 0).astype(int)
    else:
        y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def train_with_autograd(X, y, lr, n_iters):
    """独立的对照：用 PyTorch 自动求导，从 0 出发做同样步数的全量梯度下降。"""
    features = torch.tensor(X, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    for _ in range(n_iters):
        loss = F.binary_cross_entropy_with_logits(features @ w + b, targets)  # 默认对样本取平均
        grad_w, grad_b = torch.autograd.grad(loss, [w, b])
        with torch.no_grad():
            w -= lr * grad_w
            b -= lr * grad_b
    return w.detach().numpy(), b.item()


def assert_parameters(model, expected_w, expected_b, context):
    w = np.asarray(model.w, dtype=float)
    assert w.shape == expected_w.shape, f"{context}：w 的形状应为 {expected_w.shape}，实际为 {w.shape}"
    assert np.allclose(w, expected_w, atol=1e-8), f"{context}：w 不正确，最大误差 {np.abs(w - expected_w).max():.3g}"
    assert float(model.b) == pytest.approx(expected_b, abs=1e-8), f"{context}：b 不正确，应为 {expected_b:.6f}，实际为 {float(model.b):.6f}"


def test_fit_returns_self():
    X, y = random_data(0)
    model = LogisticRegression(lr=0.1, n_iters=3)
    assert model.fit(X, y) is model, "fit 应返回 self"


def test_zero_iterations_leave_parameters_at_zero():
    X, y = random_data(1)
    model = LogisticRegression(lr=0.1, n_iters=0).fit(X, y)
    assert_parameters(model, np.zeros(3), 0.0, "迭代 0 步")
    assert np.allclose(model.predict_proba(X), 0.5), "参数全为 0 时，所有样本的概率应为 0.5"


def test_first_step_by_hand():
    # 起点 w = 0、b = 0，所有预测都是 0.5。
    # 损失对 w 的梯度 = 平均( (0.5 − y) * x )，对 b 的梯度 = 平均(0.5 − y)。
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0], [-2.0, 1.0]])
    y = np.array([1, 0, 1, 1])
    model = LogisticRegression(lr=0.2, n_iters=1).fit(X, y)
    grad_w = ((0.5 - y)[:, None] * X).mean(axis=0)
    grad_b = (0.5 - y).mean()
    assert_parameters(model, -0.2 * grad_w, -0.2 * grad_b, "第一步更新之后")


@pytest.mark.parametrize("seed, n, d, lr, n_iters", [(2, 40, 3, 0.1, 1), (3, 40, 3, 0.1, 2), (4, 60, 5, 0.3, 50), (5, 25, 1, 0.05, 200), (6, 7, 4, 1.0, 30)])
def test_matches_autograd(seed, n, d, lr, n_iters):
    X, y = random_data(seed, n, d)
    model = LogisticRegression(lr=lr, n_iters=n_iters).fit(X, y)
    expected_w, expected_b = train_with_autograd(X, y, lr, n_iters)
    assert_parameters(model, expected_w, expected_b, f"与自动求导对照（{n_iters} 步）")


def test_default_hyperparameters():
    X, y = random_data(7, n=30, d=2)
    model = LogisticRegression().fit(X, y)
    expected_w, expected_b = train_with_autograd(X, y, 0.1, 1000)
    assert_parameters(model, expected_w, expected_b, "默认参数（lr=0.1，n_iters=1000）")


def test_integer_labels_and_integer_features():
    X = np.array([[1, 2], [3, -1], [0, 4], [-2, 1], [2, 2]])
    y = np.array([1, 0, 1, 1, 0])
    model = LogisticRegression(lr=0.1, n_iters=20).fit(X, y)
    expected_w, expected_b = train_with_autograd(X.astype(float), y, 0.1, 20)
    assert_parameters(model, expected_w, expected_b, "特征和标签都是整数类型")


def test_refit_starts_from_zero():
    X_a, y_a = random_data(8)
    X_b, y_b = random_data(9)
    model = LogisticRegression(lr=0.2, n_iters=25)
    model.fit(X_a, y_a)
    model.fit(X_b, y_b)
    expected_w, expected_b = train_with_autograd(X_b, y_b, 0.2, 25)
    assert_parameters(model, expected_w, expected_b, "第二次调用 fit（应重新从 0 开始）")


def test_predict_proba():
    X, y = random_data(10)
    model = LogisticRegression(lr=0.3, n_iters=40).fit(X, y)
    probabilities = np.asarray(model.predict_proba(X))
    assert probabilities.shape == (len(X),), f"predict_proba 的形状应为 {(len(X),)}，实际为 {probabilities.shape}"
    expected = np.array([1 / (1 + math.exp(-(float(np.dot(model.w, row)) + float(model.b)))) for row in X])
    assert np.allclose(probabilities, expected, atol=1e-10), "predict_proba 应等于 sigmoid(w · x + b)"


def test_predict_proba_uses_bias():
    model = LogisticRegression(lr=0.1, n_iters=1).fit(np.zeros((4, 2)), np.array([1, 1, 1, 0]))
    # 特征全为 0 时 w 不会被更新，概率完全由 b 决定
    assert float(model.b) > 0, "正样本更多、特征全为 0 时，b 应当变为正数"
    probabilities = np.asarray(model.predict_proba(np.zeros((3, 2))))
    assert np.allclose(probabilities, 1 / (1 + math.exp(-float(model.b)))), "predict_proba 没有用上偏置 b"


def test_predict_proba_extreme_inputs():
    model = LogisticRegression(lr=0.1, n_iters=1).fit(np.array([[1.0], [-1.0]]), np.array([1, 0]))
    model.w = np.array([1.0])
    model.b = 0.0
    with np.errstate(all="ignore"):
        probabilities = np.asarray(model.predict_proba(np.array([[1000.0], [-1000.0], [0.0]])))
    assert not np.isnan(probabilities).any(), "w · x + b 的绝对值很大时 predict_proba 出现了 nan"
    assert np.allclose(probabilities, [1.0, 0.0, 0.5]), f"w · x + b 为 1000、-1000、0 时概率应为 1、0、0.5，实际为 {probabilities}"


def test_predict_returns_labels():
    X, y = random_data(11)
    model = LogisticRegression(lr=0.3, n_iters=40).fit(X, y)
    predictions = np.asarray(model.predict(X))
    assert predictions.shape == (len(X),), f"predict 的形状应为 {(len(X),)}，实际为 {predictions.shape}"
    assert set(np.unique(predictions).tolist()) <= {0, 1}, "predict 应只返回 0 或 1"
    expected = (np.asarray(model.predict_proba(X)) >= 0.5).astype(int)
    assert (predictions == expected).all(), "predict 应在概率不小于 0.5 时判为 1"
    assert len(np.unique(predictions)) == 2, "这组数据上两类都应当被预测到"


def test_learns_separable_data():
    X, y = random_data(12, n=300, d=4, separable=True)
    model = LogisticRegression(lr=0.5, n_iters=2000).fit(X, y)
    accuracy = (np.asarray(model.predict(X)) == y).mean()
    assert accuracy >= 0.98, f"线性可分的数据上准确率应不低于 0.98，实际为 {accuracy:.3f}"


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_matches_reference(seed):
    X, y = random_data(seed, n=80, d=6)
    X_new, _ = random_data(seed + 100, n=30, d=6)
    model = LogisticRegression(lr=0.2, n_iters=300).fit(X, y)
    reference = Reference(lr=0.2, n_iters=300).fit(X, y)
    assert_parameters(model, reference.w, reference.b, "与参考实现对照")
    assert np.allclose(model.predict_proba(X_new), reference.predict_proba(X_new), atol=1e-10), "predict_proba 与参考实现不一致"
    assert (np.asarray(model.predict(X_new)) == reference.predict(X_new)).all(), "predict 与参考实现不一致"
