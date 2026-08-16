from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from data_encoding import apply_move, encode_fen, iccs_to_indices, indices_to_iccs
from prepare_data import convert_npz_dataset, position_value
from train import ShardDataset, append_metrics, find_shards


STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def test_position_value_uses_side_to_move_perspective() -> None:
    red_position = encode_fen(STARTING_FEN)
    start, end = iccs_to_indices("C3-C4")
    black_position = apply_move(red_position, start, end)

    assert position_value("1-0", red_position) == 1.0
    assert position_value("1-0", black_position) == -1.0
    assert position_value("0-1", red_position) == -1.0
    assert position_value("1/2-1/2", black_position) == 0.0


def test_encode_and_replay_move_updates_side_to_move() -> None:
    position = encode_fen(STARTING_FEN)
    start, end = iccs_to_indices("C3-C4")

    assert position.shape == (15, 10, 9)
    assert position.dtype == np.float32
    assert position[14].sum() == 90.0
    assert indices_to_iccs(start, end) == "C3-C4"

    next_position = apply_move(position, start, end)
    start_row, start_column = divmod(start, 9)
    end_row, end_column = divmod(end, 9)
    assert next_position[6, start_row, start_column] == 0.0
    assert next_position[6, end_row, end_column] == 1.0
    assert next_position[14].sum() == 0.0


def test_append_metrics_writes_one_json_object_per_line(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    append_metrics(metrics_path, {"epoch": 1, "training_loss": 3.5})
    append_metrics(metrics_path, {"epoch": 2, "training_loss": 3.2})

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"epoch": 1, "training_loss": 3.5},
        {"epoch": 2, "training_loss": 3.2},
    ]


def test_shard_dataset_reads_memory_mapped_batches(tmp_path: Path) -> None:
    positions = np.arange(4 * 15 * 10 * 9, dtype=np.float32).reshape(4, 15, 10, 9)
    starts = np.array([1, 2, 3, 4], dtype=np.int64)
    ends = np.array([5, 6, 7, 8], dtype=np.int64)
    np.save(tmp_path / "train-000-positions.npy", positions)
    np.save(tmp_path / "train-000-start_indices.npy", starts)
    np.save(tmp_path / "train-000-end_indices.npy", ends)
    np.save(tmp_path / "train-001-positions.npy", positions[:2])
    np.save(tmp_path / "train-001-start_indices.npy", np.array([5, 6], dtype=np.int64))
    np.save(tmp_path / "train-001-end_indices.npy", np.array([9, 10], dtype=np.int64))

    dataset = ShardDataset(find_shards(tmp_path, "train"), batch_size=3)
    batches = list(dataset)

    assert dataset.sample_count() == 6
    assert len(dataset) == 2
    assert [tuple(batch[0].shape) for batch in batches] == [(3, 15, 10, 9), (3, 15, 10, 9)]
    assert batches[0][1].tolist() == [1, 2, 3]
    assert batches[1][1].tolist() == [4, 5, 6]
    assert batches[1][2].tolist() == [8, 9, 10]

    worker_partitioned = ShardDataset(find_shards(tmp_path, "train"), batch_size=3, worker_count=2)
    assert len(worker_partitioned) == 3


def test_shard_dataset_reads_legacy_npz_batches(tmp_path: Path) -> None:
    positions = np.zeros((2, 15, 10, 9), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "train-000.npz",
        positions=positions,
        start_indices=np.array([10, 11], dtype=np.int64),
        end_indices=np.array([12, 13], dtype=np.int64),
    )

    dataset = ShardDataset(find_shards(tmp_path, "train"), batch_size=2)
    batch = next(iter(dataset))

    assert batch[0].shape == (2, 15, 10, 9)
    assert batch[1].tolist() == [10, 11]
    assert batch[2].tolist() == [12, 13]


def test_convert_npz_dataset_writes_memory_mapped_shards(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    positions = np.ones((2, 15, 10, 9), dtype=np.float32)
    np.savez_compressed(
        dataset_dir / "train-000.npz",
        positions=positions,
        start_indices=np.array([1, 2], dtype=np.int64),
        end_indices=np.array([3, 4], dtype=np.int64),
    )

    assert convert_npz_dataset(dataset_dir, overwrite=False) == 0
    converted = np.load(dataset_dir / "train-000-positions.npy", mmap_mode="r")

    assert np.array_equal(converted, positions)
    assert np.load(dataset_dir / "train-000-start_indices.npy").tolist() == [1, 2]
    assert np.load(dataset_dir / "train-000-end_indices.npy").tolist() == [3, 4]