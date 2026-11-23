"""YouTubeDNN 召回的测试。"""

import pytest
import torch

from drill.testing import assert_close, paired, randomize
from youtube_dnn import YouTubeDNN
from youtube_dnn_ref import YouTubeDNN as Reference


def by_definition(model, hist_ids, hist_mask, cand_ids):
    """按定义逐样本计算：真实历史的向量取平均 -> 用户塔 -> 与每个候选向量做点积。"""
    table = model.item_emb.weight.detach()
    rows = []
    with torch.no_grad():
        for b in range(hist_ids.size(0)):
            valid = [table[i] for i, keep in zip(hist_ids[b].tolist(), hist_mask[b].tolist()) if keep]
            average = torch.stack(valid).mean(dim=0) if valid else torch.zeros(table.size(1))
            user = model.user_tower(average)
            rows.append(torch.stack([torch.dot(user, table[c]) for c in cand_ids[b].tolist()]))
    return torch.stack(rows)


def random_batch(num_items, batch, length, k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hist_ids = torch.randint(0, num_items, (batch, length), generator=generator)
    cand_ids = torch.randint(0, num_items, (batch, k), generator=generator)
    # 每个样本的有效长度各不相同，至少有 1 个真实历史
    valid_lengths = torch.randint(1, length + 1, (batch,), generator=generator)
    hist_mask = torch.arange(length).unsqueeze(0) < valid_lengths.unsqueeze(1)
    return hist_ids, hist_mask, cand_ids


def test_parameters_follow_readme():
    paired(YouTubeDNN, Reference, 50, dim=8, hidden=16)


@pytest.mark.parametrize("batch, length, k, dim", [(5, 7, 4, 8), (1, 3, 6, 5), (4, 1, 3, 6), (3, 9, 1, 4)])
def test_matches_definition(batch, length, k, dim):
    model = randomize(YouTubeDNN(40, dim=dim, hidden=16))
    hist_ids, hist_mask, cand_ids = random_batch(40, batch, length, k)
    assert_close(model(hist_ids, hist_mask, cand_ids), by_definition(model, hist_ids, hist_mask, cand_ids), "logits")


def test_no_padding_at_all():
    model = randomize(YouTubeDNN(40, dim=6, hidden=10), seed=1)
    hist_ids, _, cand_ids = random_batch(40, 4, 5, 3, seed=1)
    hist_mask = torch.ones(4, 5, dtype=torch.bool)
    assert_close(model(hist_ids, hist_mask, cand_ids), by_definition(model, hist_ids, hist_mask, cand_ids), "logits")


def test_padding_in_the_middle_and_front():
    # padding 不一定都在末尾
    model = randomize(YouTubeDNN(40, dim=6, hidden=10), seed=2)
    hist_ids, _, cand_ids = random_batch(40, 3, 6, 4, seed=2)
    hist_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    assert_close(model(hist_ids, hist_mask, cand_ids), by_definition(model, hist_ids, hist_mask, cand_ids), "logits")


@pytest.mark.parametrize("dtype", [torch.bool, torch.long, torch.float32])
def test_mask_of_any_dtype(dtype):
    model = randomize(YouTubeDNN(40, dim=6, hidden=10), seed=3)
    hist_ids, hist_mask, cand_ids = random_batch(40, 5, 8, 3, seed=3)
    expected = by_definition(model, hist_ids, hist_mask, cand_ids)
    assert_close(model(hist_ids, hist_mask.to(dtype), cand_ids), expected, f"掩码类型为 {dtype} 时的 logits")


def test_padding_ids_do_not_matter():
    model = randomize(YouTubeDNN(40, dim=6, hidden=10), seed=4)
    hist_ids, hist_mask, cand_ids = random_batch(40, 5, 8, 3, seed=4)
    assert (~hist_mask).any(), "测试数据中应当含有 padding"
    changed = torch.where(hist_mask, hist_ids, (hist_ids + 17) % 40)
    assert_close(
        model(changed, hist_mask, cand_ids),
        model(hist_ids, hist_mask, cand_ids).detach(),
        "改动 padding 位置的 id 之后的 logits（应当不变）",
    )


def test_history_that_is_all_padding():
    model = randomize(YouTubeDNN(40, dim=6, hidden=10), seed=5)
    hist_ids, hist_mask, cand_ids = random_batch(40, 3, 5, 4, seed=5)
    hist_mask[1] = False  # 第 2 个用户没有任何真实历史
    logits = model(hist_ids, hist_mask, cand_ids)
    assert not logits.isnan().any(), "历史全是 padding 的用户出现了 nan"
    assert_close(logits, by_definition(model, hist_ids, hist_mask, cand_ids), "logits")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    model, reference = paired(YouTubeDNN, Reference, 100, dim=16, hidden=24, seed=seed)
    hist_ids, hist_mask, cand_ids = random_batch(100, 6, 10, 5, seed=seed)
    hist_mask[0] = False
    assert_close(model(hist_ids, hist_mask, cand_ids), reference(hist_ids, hist_mask, cand_ids), "logits")
