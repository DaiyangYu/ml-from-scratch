"""RQ-VAE 残差量化的测试。"""

import pytest
import torch

from rq_vae import ResidualQuantizer
from rq_vae_ref import ResidualQuantizer as Reference
from drill.testing import assert_close, paired, randomize

SHAPES = [(3, 8, 4, 5), (1, 5, 3, 2), (4, 16, 8, 1), (2, 3, 1, 6)]  # 层数, 码本大小, dim, batch


def by_definition(model, z, beta):
    """逐样本、逐层、逐码字按定义计算，返回码、各层码字之和、损失。"""
    batch, dim = z.shape
    levels = len(model.codebooks)
    codes = torch.zeros(batch, levels, dtype=torch.long)
    quantized = torch.zeros(batch, dim)
    loss = 0.0
    with torch.no_grad():
        for b in range(batch):
            residual = z[b].clone()
            for level, codebook in enumerate(model.codebooks):
                best, best_distance = -1, float("inf")
                for k in range(codebook.weight.shape[0]):
                    distance = sum((residual[i] - codebook.weight[k, i]).item() ** 2 for i in range(dim))
                    if distance < best_distance:
                        best, best_distance = k, distance
                codes[b, level] = best
                # 两项损失数值相同，合起来是 (1 + beta) 倍；mse 对全部 batch × dim 个元素取平均
                loss += (1 + beta) * best_distance / (batch * dim)
                quantized[b] += codebook.weight[best]
                residual = residual - codebook.weight[best]
    return codes, quantized, torch.tensor(loss)


def expected_gradients(model, z, codes, beta):
    """按定义推出的损失梯度：对 z 的梯度，以及对每层码本的梯度。"""
    batch, dim = z.shape
    scale = 2.0 / (batch * dim)
    grad_z = torch.zeros_like(z)
    grad_codebooks = []
    with torch.no_grad():
        residual = z.clone()
        for level, codebook in enumerate(model.codebooks):
            e = codebook.weight[codes[:, level]]
            grad_z += beta * scale * (residual - e)  # 只有 commitment loss 对 z 有梯度
            grad = torch.zeros_like(codebook.weight)
            grad.index_add_(0, codes[:, level], scale * (e - residual))  # 只有本层的前一项对码本有梯度
            grad_codebooks.append(grad)
            residual = residual - e
    return grad_z, grad_codebooks


def build(levels, size, dim, beta=0.25, seed=0):
    return randomize(ResidualQuantizer(levels, size, dim, beta=beta), seed=seed)


def test_parameters_follow_readme():
    paired(ResidualQuantizer, Reference, 3, 8, 4)
    paired(ResidualQuantizer, Reference, 1, 5, 2, beta=0.5)


@pytest.mark.parametrize("levels, size, dim, batch", SHAPES)
def test_output_shapes_and_types(levels, size, dim, batch):
    torch.manual_seed(0)
    model = build(levels, size, dim)
    z_q, loss, codes = model(torch.randn(batch, dim))
    assert tuple(z_q.shape) == (batch, dim), f"z_q 的形状应为 {(batch, dim)}，实际为 {tuple(z_q.shape)}"
    assert loss.dim() == 0, f"loss 应为标量张量，实际形状为 {tuple(loss.shape)}"
    assert tuple(codes.shape) == (batch, levels), f"codes 的形状应为 {(batch, levels)}，实际为 {tuple(codes.shape)}"
    assert codes.dtype == torch.long, f"codes 应为 torch.long，实际为 {codes.dtype}"
    assert int(codes.min()) >= 0 and int(codes.max()) < size, "codes 中出现了超出码本范围的编号"


@pytest.mark.parametrize("levels, size, dim, batch", SHAPES)
@pytest.mark.parametrize("beta", [0.25, 0.6])
def test_matches_definition(levels, size, dim, batch, beta):
    torch.manual_seed(1)
    model = build(levels, size, dim, beta=beta, seed=1)
    z = torch.randn(batch, dim)
    expected_codes, expected_quantized, expected_loss = by_definition(model, z, beta)
    with torch.no_grad():
        z_q, loss, codes = model(z)
    assert torch.equal(codes, expected_codes), f"codes 不正确，应为\n{expected_codes}\n实际为\n{codes}"
    assert_close(z_q, expected_quantized, "z_q 的数值（应为各层码字之和）")
    assert_close(loss, expected_loss, "loss")


def test_default_beta_is_a_quarter():
    torch.manual_seed(2)
    model = randomize(ResidualQuantizer(3, 8, 4), seed=2)
    z = torch.randn(5, 4)
    _, _, expected_loss = by_definition(model, z, 0.25)
    with torch.no_grad():
        assert_close(model(z)[1], expected_loss, "beta 取默认值时的 loss")


def test_later_levels_quantize_the_residual():
    # 第一层码本里恰好有 z 本身，残差为 0；第二层应当选离零向量最近的码字，而不是离 z 最近的
    model = ResidualQuantizer(2, 2, 2)
    with torch.no_grad():
        model.codebooks[0].weight.copy_(torch.tensor([[3.0, 4.0], [-3.0, -4.0]]))
        model.codebooks[1].weight.copy_(torch.tensor([[3.0, 3.0], [0.1, 0.0]]))
        z_q, loss, codes = model(torch.tensor([[3.0, 4.0]]))
    assert codes.tolist() == [[0, 1]], f"codes 应为 [[0, 1]]，实际为 {codes.tolist()}"
    assert_close(z_q, torch.tensor([[3.1, 4.0]]), "z_q")
    # 第一层残差为 0，没有损失；第二层的误差平方和是 0.01，除以元素个数 2，再乘 (1 + 0.25)
    assert_close(loss, torch.tensor(1.25 * 0.01 / 2), "loss", atol=1e-7)


def test_each_level_uses_its_own_codebook():
    torch.manual_seed(3)
    model = build(3, 6, 4, seed=3)
    z = torch.randn(7, 4)
    with torch.no_grad():
        z_q, _, codes = model(z)
        total = sum(model.codebooks[level].weight[codes[:, level]] for level in range(3))
    assert_close(z_q, total, "z_q（应等于按 codes 在各层码本中查出的码字之和）")


@pytest.mark.parametrize("levels, size, dim, batch", SHAPES)
def test_straight_through_gradient(levels, size, dim, batch):
    torch.manual_seed(4)
    model = build(levels, size, dim, seed=4)
    z = torch.randn(batch, dim, requires_grad=True)
    weights = torch.randn(batch, dim)
    z_q, _, _ = model(z)
    assert z_q.requires_grad, "z_q 没有把梯度传回 z（缺少直通估计）"
    parameters = [codebook.weight for codebook in model.codebooks]
    grads = torch.autograd.grad((z_q * weights).sum(), [z] + parameters, allow_unused=True)
    assert grads[0] is not None, "z_q 没有把梯度传回 z（缺少直通估计）"
    assert_close(grads[0], weights, "z_q 对 z 的梯度（应等同于恒等映射）")
    for level, grad in enumerate(grads[1:]):
        assert grad is None or float(grad.abs().max()) == 0.0, f"z_q 不应向第 {level + 1} 层码本传梯度"


@pytest.mark.parametrize("levels, size, dim, batch", SHAPES)
@pytest.mark.parametrize("beta", [0.25, 0.6])
def test_loss_gradients(levels, size, dim, batch, beta):
    torch.manual_seed(5)
    model = build(levels, size, dim, beta=beta, seed=5)
    z = torch.randn(batch, dim, requires_grad=True)
    _, loss, codes = model(z)
    parameters = [codebook.weight for codebook in model.codebooks]
    grads = torch.autograd.grad(loss, [z] + parameters, allow_unused=True)
    expected_z, expected_codebooks = expected_gradients(model, z.detach(), codes, beta)
    assert grads[0] is not None, "loss 对 z 没有梯度（commitment loss 缺失或被 detach）"
    assert_close(grads[0], expected_z, "loss 对 z 的梯度", atol=1e-6)
    for level, (grad, expected) in enumerate(zip(grads[1:], expected_codebooks)):
        assert grad is not None, f"loss 对第 {level + 1} 层码本没有梯度"
        assert_close(grad, expected, f"loss 对第 {level + 1} 层码本的梯度", atol=1e-6)


@pytest.mark.parametrize("levels, size, dim, batch", SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_reference(levels, size, dim, batch, seed):
    torch.manual_seed(seed)
    model, reference = paired(ResidualQuantizer, Reference, levels, size, dim, beta=0.4, seed=seed)
    z = torch.randn(batch, dim)
    with torch.no_grad():
        z_q, loss, codes = model(z)
        expected_z_q, expected_loss, expected_codes = reference(z)
    assert torch.equal(codes, expected_codes), "codes 与参考实现不一致"
    assert_close(z_q, expected_z_q, "z_q")
    assert_close(loss, expected_loss, "loss")
