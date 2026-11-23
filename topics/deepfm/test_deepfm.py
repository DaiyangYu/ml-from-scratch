"""DeepFM 的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from deepfm import DeepFM
from deepfm_ref import DeepFM as Reference


def fm_by_definition(model, ids):
    """单个样本的 FM 部分，按定义计算：w0 + Σ w_i + Σ_{i<j} <v_i, v_j>。"""
    y = model.w0.detach().clone().squeeze()
    for i in ids:
        y = y + model.linear.weight[i, 0]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            y = y + torch.dot(model.emb.weight[ids[a]], model.emb.weight[ids[b]])
    return y


def dnn_by_hand(model, ids):
    """单个样本的 DNN 部分：各特征域的 embedding 依次拼接后过两层感知机。"""
    first, second = model.dnn[0], model.dnn[2]
    dnn_in = torch.cat([model.emb.weight[i] for i in ids])
    hidden = torch.clamp(first.weight @ dnn_in + first.bias, min=0)
    return second.weight[0] @ hidden + second.bias[0]


def by_definition(model, feat_ids):
    rows = [fm_by_definition(model, ids) + dnn_by_hand(model, ids) for ids in feat_ids.tolist()]
    return torch.stack(rows).unsqueeze(-1).detach()


def test_parameters_follow_readme():
    # 参数名检查同时保证只有一张 Embedding 表，FM 分支和 DNN 分支没有各建一张
    paired(DeepFM, Reference, 50, 5, dim=8, hidden=12)


def test_w0_starts_at_zero():
    assert DeepFM(10, 3, dim=4, hidden=6).w0.detach().abs().sum().item() == 0


@pytest.mark.parametrize(
    "batch, num_features, n_fields, dim, hidden",
    [
        (4, 30, 5, 8, 12),
        (1, 100, 2, 3, 4),
        (3, 6, 9, 1, 5),   # 特征域个数大于特征 id 总数，隐向量只有 1 维
        (2, 20, 1, 4, 6),  # 只有一个特征域，没有交叉项
    ],
)
def test_matches_definition(batch, num_features, n_fields, dim, hidden):
    torch.manual_seed(0)
    model = randomize(DeepFM(num_features, n_fields, dim=dim, hidden=hidden))
    feat_ids = torch.randint(0, num_features, (batch, n_fields))
    assert_close(model(feat_ids), by_definition(model, feat_ids), atol=1e-4)


def test_repeated_feature_in_one_sample():
    model = randomize(DeepFM(30, 4, dim=6, hidden=8), seed=1)
    feat_ids = torch.tensor([[3, 3, 7, 3]])
    assert_close(model(feat_ids), by_definition(model, feat_ids), atol=1e-4)


def test_both_branches_read_the_same_embedding_table():
    # 只改动 emb 的权重：FM 的交叉项和 DNN 的输出都应随之变化，变化量与按定义计算的一致
    torch.manual_seed(2)
    model = randomize(DeepFM(20, 4, dim=5, hidden=8), seed=2)
    feat_ids = torch.randint(0, 20, (3, 4))
    with torch.no_grad():
        model.emb.weight.mul_(-0.7).add_(0.3)
    assert_close(model(feat_ids), by_definition(model, feat_ids), "改动 emb 之后的输出", atol=1e-4)


def test_field_order_matters_only_for_dnn():
    # 交换两个特征域：FM 部分不变，DNN 部分的输入顺序变了，整体输出按定义重新计算
    torch.manual_seed(3)
    model = randomize(DeepFM(20, 3, dim=4, hidden=6), seed=3)
    feat_ids = torch.tensor([[1, 7, 12], [5, 5, 9]])
    swapped = feat_ids[:, [2, 1, 0]]
    assert_close(model(swapped), by_definition(model, swapped), "交换特征域之后的输出", atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    torch.manual_seed(seed)
    model, reference = paired(DeepFM, Reference, 100, 10, dim=16, hidden=64, seed=seed)
    feat_ids = torch.randint(0, 100, (8, 10))
    assert_close(model(feat_ids), reference(feat_ids), atol=1e-4)
