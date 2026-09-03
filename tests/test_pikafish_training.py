from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def _write_shard(dataset: Path) -> None:
    prefix = dataset / "train-000"
    np.save(f"{prefix}-positions.npy", np.zeros((3, 15, 10, 9), dtype=np.float32))
    np.save(f"{prefix}-teacher_score_kinds.npy", np.asarray([0, 1, 0], dtype=np.uint8))
    np.save(f"{prefix}-teacher_scores.npy", np.asarray([100, 2, -50], dtype=np.float32))
    np.save(f"{prefix}-candidate_action_ids.npy", np.asarray([[1, 2, -1], [4, 5, -1], [9, -1, -1]], dtype=np.int16))
    np.save(f"{prefix}-candidate_score_kinds.npy", np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8))
    np.save(f"{prefix}-candidate_scores.npy", np.asarray([[100, 0, 0], [2, -20, 0], [-50, 0, 0]], dtype=np.float32))
    np.save(f"{prefix}-legal_action_ids.npy", np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int16))
    np.save(f"{prefix}-legal_action_offsets.npy", np.asarray([0, 3, 5, 6], dtype=np.int64))


def test_pikafish_dataset_and_typed_losses(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from train_pikafish import PikafishShardDataset, collate_pikafish, compute_losses

    _write_shard(tmp_path)
    dataset = PikafishShardDataset(tmp_path, "train")
    batch = collate_pikafish([dataset[index] for index in range(len(dataset))])
    policy_logits = torch.zeros((3, 8100), requires_grad=True)
    predictions = torch.zeros((3, 1), requires_grad=True)
    losses = compute_losses(policy_logits, predictions, batch, temperature=50, value_scale=400)
    (losses["policy_loss"] + losses["value_loss"]).backward()

    assert len(dataset) == 3
    assert losses["policy_valid"].tolist() == [True, True, False]
    assert losses["mate_policy"].tolist() == [False, True, False]
    assert losses["value_valid"].tolist() == [True, True, True]
    assert losses["value_mate"].tolist() == [False, True, False]
    assert losses["cp_policy_count"].item() == 1
    assert losses["mate_policy_count"].item() == 1
    assert losses["value_cp_count"].item() == 2
    assert losses["value_mate_count"].item() == 1
    assert policy_logits.grad is not None
    assert predictions.grad is not None


def test_block_shuffle_sampler_visits_every_index_once(tmp_path: Path) -> None:
    from train_pikafish import BlockShuffleSampler, PikafishShardDataset

    _write_shard(tmp_path)
    dataset = PikafishShardDataset(tmp_path, "train")
    sampler = BlockShuffleSampler(dataset, block_size=2, seed=7)
    sampler.set_epoch(3)

    assert sorted(sampler) == [0, 1, 2]


def test_horizontal_mirror_transforms_positions_and_all_action_sets(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from train_pikafish import PikafishShardDataset, collate_pikafish, with_horizontal_mirror

    _write_shard(tmp_path)
    batch = collate_pikafish([PikafishShardDataset(tmp_path, "train")[0]])
    batch["positions"][0, 0, 0, 0] = 1
    mirrored = with_horizontal_mirror(batch)

    assert mirrored["positions"].shape[0] == 2
    assert mirrored["positions"][1, 0, 0, 8] == 1
    assert mirrored["candidate_action_ids"].tolist() == [[1, 2, -1], [727, 726, -1]]
    assert mirrored["legal_action_ids"].tolist() == [1, 2, 3, 727, 726, 725]
    assert mirrored["legal_action_offsets"].tolist() == [0, 3, 6]