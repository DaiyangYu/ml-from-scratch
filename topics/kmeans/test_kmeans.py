"""K-means 的测试。"""

import numpy as np
import pytest

from kmeans import kmeans
from kmeans_ref import kmeans as reference


def random_data(seed, n=40, d=3, k=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * 2
    init = X[rng.choice(n, size=k, replace=False)].copy()
    return X, init


def nearest(point, centers):
    """逐个中心比较，返回最近中心的编号；距离相同时取编号最小的。"""
    best, best_distance = -1, float("inf")
    for j, center in enumerate(centers):
        distance = sum((float(a) - float(b)) ** 2 for a, b in zip(point, center))
        if distance < best_distance:
            best, best_distance = j, distance
    return best


def by_definition(X, init_centers, max_iters):
    """独立的对照：全部用循环按定义写，固定做满 max_iters 轮。"""
    centers = [[float(v) for v in row] for row in init_centers]
    dims = len(centers[0])
    for _ in range(max_iters):
        labels = [nearest(point, centers) for point in X]
        for j in range(len(centers)):
            members = [X[i] for i in range(len(X)) if labels[i] == j]
            if members:
                centers[j] = [sum(float(m[t]) for m in members) / len(members) for t in range(dims)]
    labels = [nearest(point, centers) for point in X]
    return np.array(centers), np.array(labels)


def inertia(X, centers, labels):
    return float(((np.asarray(X, dtype=float) - np.asarray(centers)[np.asarray(labels)]) ** 2).sum())


def assert_result(result, expected_centers, expected_labels, context):
    assert isinstance(result, tuple) and len(result) == 2, f"{context}：应返回 (centers, labels)"
    centers, labels = np.asarray(result[0]), np.asarray(result[1])
    assert centers.shape == expected_centers.shape, f"{context}：centers 的形状应为 {expected_centers.shape}，实际为 {centers.shape}"
    assert labels.shape == expected_labels.shape, f"{context}：labels 的形状应为 {expected_labels.shape}，实际为 {labels.shape}"
    assert not np.isnan(centers.astype(float)).any(), f"{context}：centers 中出现了 nan"
    assert np.allclose(centers, expected_centers, atol=1e-9), (
        f"{context}：centers 不正确，最大误差 {np.abs(centers - expected_centers).max():.3g}"
    )
    assert np.issubdtype(labels.dtype, np.integer), f"{context}：labels 应为整数数组，实际为 {labels.dtype}"
    assert (labels == expected_labels).all(), f"{context}：labels 不正确，应为 {expected_labels.tolist()}，实际为 {labels.tolist()}"


def test_zero_iterations_only_assign():
    X, init = random_data(0)
    expected_centers, expected_labels = by_definition(X, init, 0)
    assert_result(kmeans(X, init, max_iters=0), expected_centers, expected_labels, "max_iters=0")
    assert np.allclose(expected_centers, init)


def test_one_iteration_by_hand():
    # 一维的 6 个点，初始中心 0 和 1。
    # 第一轮分配：0 归中心 0，其余 5 个点归中心 1；更新后中心为 0 和 (1+2+10+11+12)/5 = 7.2。
    # 按最终中心重新分配：0、1、2 归中心 0，10、11、12 归中心 1。
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    init = np.array([[0.0], [1.0]])
    assert_result(kmeans(X, init, max_iters=1), np.array([[0.0], [7.2]]), np.array([0, 0, 0, 1, 1, 1]), "手算的第一轮")


def test_converged_by_hand():
    # 接上例：第二轮更新后中心为 1 和 11，之后不再变化。
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    init = np.array([[0.0], [1.0]])
    expected = (np.array([[1.0], [11.0]]), np.array([0, 0, 0, 1, 1, 1]))
    assert_result(kmeans(X, init, max_iters=2), *expected, "手算的第二轮")
    assert_result(kmeans(X, init), *expected, "默认 max_iters（收敛后）")


@pytest.mark.parametrize("max_iters", [1, 2, 3, 5, 50])
@pytest.mark.parametrize("seed, n, d, k", [(1, 40, 3, 4), (2, 60, 2, 5), (3, 25, 1, 3), (4, 30, 6, 2), (5, 12, 4, 1)])
def test_matches_definition(seed, n, d, k, max_iters):
    X, init = random_data(seed, n, d, k)
    expected_centers, expected_labels = by_definition(X, init, max_iters)
    assert_result(kmeans(X, init, max_iters=max_iters), expected_centers, expected_labels, f"与按定义的循环实现对照（{max_iters} 轮）")


def test_default_max_iters_is_100():
    # 沿一条线均匀排开的点收敛很慢，用来区分默认轮数；先确认这组数据在 100 轮内确实还没有收敛
    X = np.linspace(0.0, 1.0, 400).reshape(-1, 1) ** 3
    init = np.linspace(0.0, 0.01, 12).reshape(-1, 1)
    at_100, labels_100 = by_definition(X, init, 100)
    at_99, _ = by_definition(X, init, 99)
    at_101, _ = by_definition(X, init, 101)
    if np.allclose(at_99, at_100, atol=1e-9) or np.allclose(at_100, at_101, atol=1e-9):
        pytest.skip("这组数据在 100 轮内已收敛，无法区分默认轮数")
    assert_result(kmeans(X, init), at_100, labels_100, "默认 max_iters 应为 100")


def test_empty_cluster_keeps_its_center():
    X = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0]])
    init = np.array([[0.0, 1.0], [100.0, 100.0], [4.0, 1.0]])  # 中间这个中心离所有点都很远，分不到样本
    with np.errstate(all="ignore"):
        result = kmeans(X, init, max_iters=10)
    assert_result(result, init, np.array([0, 0, 2, 2]), "有空簇时")


def test_tie_goes_to_lowest_index():
    X = np.array([[0.0], [1.0], [-3.0], [5.0]])
    init = np.array([[-1.0], [3.0]])  # 点 1 到两个中心的距离相同
    assert_result(kmeans(X, init, max_iters=0), init, np.array([0, 0, 0, 1]), "距离相同时")


def test_integer_inputs():
    X = np.array([[0, 0], [1, 0], [0, 1], [9, 9], [10, 9], [9, 10]])
    init = np.array([[0, 0], [1, 0]])
    expected_centers, expected_labels = by_definition(X, init, 20)
    result = kmeans(X, init, max_iters=20)
    assert np.issubdtype(np.asarray(result[0]).dtype, np.floating), "centers 应为浮点数组（整数输入时均值会被截断）"
    assert_result(result, expected_centers, expected_labels, "输入是整数类型")
    assert np.allclose(expected_centers, [[1 / 3, 1 / 3], [28 / 3, 28 / 3]])


def test_inputs_are_not_modified():
    X, init = random_data(6)
    X_before, init_before = X.copy(), init.copy()
    kmeans(X, init, max_iters=10)
    assert np.array_equal(X, X_before), "X 被改动了"
    assert np.array_equal(init, init_before), "init_centers 被改动了"


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_converged_centers_are_cluster_means(seed):
    X, init = random_data(seed, n=80, d=2, k=4)
    centers, labels = kmeans(X, init, max_iters=1000)
    centers, labels = np.asarray(centers), np.asarray(labels)
    for j in range(4):
        members = X[labels == j]
        if len(members):
            assert np.allclose(centers[j], members.mean(axis=0), atol=1e-9), f"收敛后第 {j} 个中心应等于簇内样本的均值"


@pytest.mark.parametrize("seed", [10, 11])
def test_inertia_never_increases(seed):
    X, init = random_data(seed, n=100, d=3, k=5)
    values = [inertia(X, *kmeans(X, init, max_iters=t)) for t in range(8)]
    for earlier, later in zip(values, values[1:]):
        assert later <= earlier + 1e-9, f"轮数增加后簇内平方误差之和变大了：{values}"
    assert values[-1] < values[0], "迭代之后簇内平方误差之和应当下降"


def test_recovers_separated_blobs():
    rng = np.random.default_rng(12)
    true_centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.concatenate([center + rng.normal(size=(50, 2)) * 0.5 for center in true_centers])
    init = X[[0, 50, 100]]
    centers, labels = kmeans(X, init)
    assert np.allclose(np.asarray(centers), true_centers, atol=0.3), "三个分得很开的簇的中心没有找对"
    assert (np.asarray(labels) == np.repeat([0, 1, 2], 50)).all(), "三个分得很开的簇的分配结果不正确"


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_matches_reference(seed):
    X, init = random_data(seed, n=120, d=5, k=6)
    for max_iters in (1, 4, 100):
        expected_centers, expected_labels = reference(X, init, max_iters=max_iters)
        assert_result(kmeans(X, init, max_iters=max_iters), expected_centers, expected_labels, f"与参考实现对照（{max_iters} 轮）")
