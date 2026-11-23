"""各题目测试共用的辅助函数。

测试的基本做法：让待测模块与参考实现持有同一组随机权重，再比较二者的输出。
因此题目的 README 会规定参数的名字，写法可以不同，但参数名和形状必须一致。
"""

from __future__ import annotations

import torch
from torch import nn


def randomize(module: nn.Module, seed: int = 0, scale: float = 0.5) -> nn.Module:
    """把全部参数设为随机值。偏置等参数默认初始化为 0，不随机化就会掩盖漏加偏置之类的错误。"""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)
    return module


def assert_same_parameters(model: nn.Module, reference: nn.Module) -> None:
    """检查参数名和形状是否符合 README 的规定。"""
    mine = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    expected = {name: tuple(value.shape) for name, value in reference.state_dict().items()}
    problems = []
    for name in sorted(expected.keys() - mine.keys()):
        problems.append(f"缺少参数 {name}，形状应为 {expected[name]}")
    for name in sorted(mine.keys() - expected.keys()):
        problems.append(f"多出参数 {name}")
    for name in sorted(expected.keys() & mine.keys()):
        if mine[name] != expected[name]:
            problems.append(f"参数 {name} 的形状应为 {expected[name]}，实际为 {mine[name]}")
    assert not problems, "参数与 README 的规定不一致：\n  " + "\n  ".join(problems)


def paired(model_class, reference_class, *args, seed: int = 0, **kwargs):
    """用同样的构造参数建立待测模块和参考模块，并让二者持有同一组随机权重。"""
    model = model_class(*args, **kwargs)
    reference = reference_class(*args, **kwargs)
    assert_same_parameters(model, reference)
    randomize(reference, seed)
    model.load_state_dict(reference.state_dict())
    return model.eval(), reference.eval()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str = "输出", atol: float = 1e-5) -> None:
    assert isinstance(actual, torch.Tensor), f"{what}应为张量，实际为 {type(actual).__name__}"
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"{what}的形状应为 {tuple(expected.shape)}，实际为 {tuple(actual.shape)}"
    )
    assert not actual.isnan().any(), f"{what}中出现了 NaN"
    error = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol, rtol=1e-4), f"{what}不正确，最大误差 {error:.3g}"
