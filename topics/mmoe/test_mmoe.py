"""MMoE 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from mmoe import MMoE
from mmoe_ref import MMoE as Reference


def by_formula(model, x):
    """按论文公式逐样本计算：第 k 个任务的表示 = Σ_e softmax(W_gk x)_e · 专家_e(x)。"""
    outputs = []
    for gate in model.gates:
        rows = []
        for sample in x:
            expert_outputs = []
            for expert in model.experts:
                first, second = expert[0], expert[2]
                hidden = torch.clamp(first.weight @ sample + first.bias, min=0)
                expert_outputs.append(second.weight @ hidden + second.bias)
            logits = gate.weight @ sample + gate.bias
            weights = logits.exp() / logits.exp().sum()
            rows.append(sum(w * out for w, out in zip(weights, expert_outputs)))
        outputs.append(torch.stack(rows).detach())
    return outputs


def check_outputs(actual, expected):
    assert isinstance(actual, (list, tuple)), f"forward 应返回列表，实际为 {type(actual).__name__}"
    assert len(actual) == len(expected), f"返回的列表长度应等于任务数 {len(expected)}，实际为 {len(actual)}"
    for k, (mine, theirs) in enumerate(zip(actual, expected)):
        assert_close(mine, theirs, f"第 {k} 个任务的表示")


def test_parameters_follow_readme():
    paired(MMoE, Reference, 7, 5, num_experts=3, num_tasks=4)


@pytest.mark.parametrize(
    "batch, in_dim, expert_dim, num_experts, num_tasks",
    [
        (4, 6, 5, 3, 2),
        (3, 4, 4, 4, 3),  # 专家输出维度与专家个数相等
        (2, 5, 3, 6, 1),  # 只有一个任务
        (5, 3, 4, 1, 2),  # 只有一个专家
    ],
)
def test_matches_formula(batch, in_dim, expert_dim, num_experts, num_tasks):
    torch.manual_seed(0)
    model = randomize(MMoE(in_dim, expert_dim, num_experts=num_experts, num_tasks=num_tasks))
    x = torch.randn(batch, in_dim)
    check_outputs(model(x), by_formula(model, x))


def test_each_task_uses_its_own_gate():
    torch.manual_seed(1)
    model = randomize(MMoE(6, 5, num_experts=4, num_tasks=3), seed=1)
    x = torch.randn(4, 6)
    outputs = model(x)
    check_outputs(outputs, by_formula(model, x))
    assert not torch.allclose(outputs[0], outputs[1]), "不同任务的 gate 不同，得到的表示不应相同"
    assert not torch.allclose(outputs[1], outputs[2]), "不同任务的 gate 不同，得到的表示不应相同"


def test_identical_experts_give_expert_output():
    # 各专家权重完全相同时，无论 gate 怎么分配，加权结果都应等于单个专家的输出
    torch.manual_seed(2)
    model = randomize(MMoE(5, 4, num_experts=3, num_tasks=2), seed=2)
    for expert in model.experts[1:]:
        expert.load_state_dict(model.experts[0].state_dict())
    x = torch.randn(6, 5)
    single = model.experts[0](x).detach()
    for k, output in enumerate(model(x)):
        assert_close(output, single, f"专家完全相同时第 {k} 个任务的表示（各专家权重之和应为 1）")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(MMoE, Reference, 10, 8, num_experts=4, num_tasks=3, seed=seed)
    x = torch.randn(7, 10)
    check_outputs(model(x), [out.detach() for out in reference(x)])
