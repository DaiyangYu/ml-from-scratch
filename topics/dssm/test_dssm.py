"""DSSM 双塔的测试。"""

import pytest
import torch
import torch.nn.functional as F

from drill.testing import assert_close, paired, randomize
from dssm import DSSM
from dssm_ref import DSSM as Reference


def tower_by_hand(tower, x):
    """按 README 规定的结构手工展开一个塔：Linear -> ReLU -> Linear。"""
    state = {name: value.detach() for name, value in tower.state_dict().items()}
    hidden = torch.relu(x @ state["0.weight"].T + state["0.bias"])
    return hidden @ state["2.weight"].T + state["2.bias"]


def random_inputs(batch, user_dim, item_dim, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, user_dim, generator=generator), torch.randn(batch, item_dim, generator=generator)


def test_parameters_follow_readme():
    paired(DSSM, Reference, 10, 7, out_dim=8, hidden=16)


@pytest.mark.parametrize("batch, user_dim, item_dim, out_dim", [(5, 10, 7, 8), (1, 3, 12, 4), (6, 9, 9, 2)])
def test_encoded_vectors_have_unit_length(batch, user_dim, item_dim, out_dim):
    model = randomize(DSSM(user_dim, item_dim, out_dim=out_dim, hidden=16))
    user_feats, item_feats = random_inputs(batch, user_dim, item_dim)
    u, i = model.encode_user(user_feats), model.encode_item(item_feats)
    assert tuple(u.shape) == (batch, out_dim), f"用户向量的形状应为 {(batch, out_dim)}，实际为 {tuple(u.shape)}"
    assert tuple(i.shape) == (batch, out_dim), f"物品向量的形状应为 {(batch, out_dim)}，实际为 {tuple(i.shape)}"
    assert_close(u.norm(dim=-1), torch.ones(batch), "用户向量的长度（每一行都应为 1）")
    assert_close(i.norm(dim=-1), torch.ones(batch), "物品向量的长度（每一行都应为 1）")


def test_encoders_match_hand_expansion():
    model = randomize(DSSM(10, 7, out_dim=8, hidden=16), seed=1)
    user_feats, item_feats = random_inputs(5, 10, 7, seed=1)
    raw_user = tower_by_hand(model.user_tower, user_feats)
    raw_item = tower_by_hand(model.item_tower, item_feats)
    expected_user = raw_user / raw_user.norm(dim=-1, keepdim=True)
    expected_item = raw_item / raw_item.norm(dim=-1, keepdim=True)
    assert_close(model.encode_user(user_feats), expected_user, "用户向量")
    assert_close(model.encode_item(item_feats), expected_item, "物品向量")


@pytest.mark.parametrize("batch, user_dim, item_dim, out_dim", [(5, 10, 7, 8), (1, 3, 12, 4), (6, 9, 9, 2)])
def test_forward_is_cosine_similarity(batch, user_dim, item_dim, out_dim):
    model = randomize(DSSM(user_dim, item_dim, out_dim=out_dim, hidden=16), seed=2)
    user_feats, item_feats = random_inputs(batch, user_dim, item_dim, seed=2)
    expected = F.cosine_similarity(
        tower_by_hand(model.user_tower, user_feats), tower_by_hand(model.item_tower, item_feats), dim=-1
    )
    scores = model(user_feats, item_feats)
    assert_close(scores, expected, "匹配分")
    assert scores.abs().max().item() <= 1 + 1e-5, "余弦相似度应落在 [-1, 1] 之内"


def test_forward_agrees_with_encoders():
    model = randomize(DSSM(10, 7, out_dim=8, hidden=16), seed=3)
    user_feats, item_feats = random_inputs(4, 10, 7, seed=3)
    expected = (model.encode_user(user_feats) * model.encode_item(item_feats)).sum(dim=-1)
    assert_close(model(user_feats, item_feats), expected.detach(), "匹配分（应等于两个编码向量的逐行点积）")


def test_identical_towers_and_inputs_give_one():
    # 两个塔权重相同、输入相同时，两个向量方向完全一致，余弦相似度为 1
    model = randomize(DSSM(6, 6, out_dim=5, hidden=9), seed=4)
    model.item_tower.load_state_dict(model.user_tower.state_dict())
    feats, _ = random_inputs(3, 6, 6, seed=4)
    assert_close(model(feats, feats), torch.ones(3), "相同向量之间的匹配分")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    model, reference = paired(DSSM, Reference, 12, 20, out_dim=16, hidden=24, seed=seed)
    user_feats, item_feats = random_inputs(7, 12, 20, seed=seed)
    assert_close(model.encode_user(user_feats), reference.encode_user(user_feats), "用户向量")
    assert_close(model.encode_item(item_feats), reference.encode_item(item_feats), "物品向量")
    assert_close(model(user_feats, item_feats), reference(user_feats, item_feats), "匹配分")
