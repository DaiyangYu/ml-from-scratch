"""Skip-gram + 负采样的测试。"""

import math

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from skipgram import SkipGramNS
from skipgram_ref import SkipGramNS as Reference


def log_sigmoid(x):
    return -math.log1p(math.exp(-x))


def by_definition(model, center, pos_context, neg_contexts):
    """按原论文的目标函数逐样本逐项计算：-log σ(u·v_pos) - Σ_k log σ(-u·v_neg_k)，再对 batch 取平均。"""
    center_table = model.center_emb.weight.detach()
    context_table = model.context_emb.weight.detach()
    total = 0.0
    for b in range(center.size(0)):
        u = center_table[center[b]]
        loss = -log_sigmoid(torch.dot(u, context_table[pos_context[b]]).item())
        for k in range(neg_contexts.size(1)):
            loss -= log_sigmoid(-torch.dot(u, context_table[neg_contexts[b, k]]).item())
        total += loss
    return torch.tensor(total / center.size(0))


def random_batch(vocab_size, batch, k):
    return (
        torch.randint(0, vocab_size, (batch,)),
        torch.randint(0, vocab_size, (batch,)),
        torch.randint(0, vocab_size, (batch, k)),
    )


def test_parameters_follow_readme():
    paired(SkipGramNS, Reference, 50, dim=8)


def test_loss_is_scalar():
    model = randomize(SkipGramNS(50, dim=8))
    loss = model(*random_batch(50, 4, 3))
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "损失应为 0 维的标量张量"


def test_all_zero_embeddings():
    # 所有向量为 0 时每一项得分都是 0，每一项损失都是 ln 2，每个样本共 1 + K 项
    model = SkipGramNS(20, dim=4)
    with torch.no_grad():
        model.center_emb.weight.zero_()
        model.context_emb.weight.zero_()
    loss = model(*random_batch(20, 5, 3))
    assert_close(loss, torch.tensor((1 + 3) * math.log(2.0)), "全零向量时的损失（应为 (1 + K) · ln 2）")


def test_hand_computed_values():
    model = SkipGramNS(3, dim=2)
    with torch.no_grad():
        model.center_emb.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]))
        model.context_emb.weight.copy_(torch.tensor([[2.0, 5.0], [-1.0, 1.0], [0.5, -3.0]]))
    # 样本 0：中心词 0，正样本 0（得分 2），负样本 1、2（得分 -1、0.5）
    # 样本 1：中心词 1，正样本 1（得分 2），负样本 2、0（得分 -6、10）
    loss = model(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[1, 2], [2, 0]]))
    sample_0 = -log_sigmoid(2.0) - log_sigmoid(1.0) - log_sigmoid(-0.5)
    sample_1 = -log_sigmoid(2.0) - log_sigmoid(6.0) - log_sigmoid(-10.0)
    assert_close(loss, torch.tensor((sample_0 + sample_1) / 2), "损失")


@pytest.mark.parametrize("batch, k, dim", [(6, 5, 8), (1, 4, 3), (4, 1, 6), (3, 7, 1)])
def test_matches_definition(batch, k, dim):
    torch.manual_seed(0)
    model = randomize(SkipGramNS(30, dim=dim))
    inputs = random_batch(30, batch, k)
    assert_close(model(*inputs), by_definition(model, *inputs), "损失")


def test_same_word_as_center_and_context():
    # 同一个词作中心词和作上下文词时，要分别从两张表里取向量
    model = randomize(SkipGramNS(10, dim=5), seed=3)
    inputs = (torch.tensor([4, 4]), torch.tensor([4, 2]), torch.tensor([[4, 1, 4], [0, 4, 9]]))
    assert_close(model(*inputs), by_definition(model, *inputs), "损失")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(SkipGramNS, Reference, 80, dim=16, seed=seed)
    inputs = random_batch(80, 7, 5)
    assert_close(model(*inputs), reference(*inputs), "损失")
