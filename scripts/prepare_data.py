from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from data_utils import BLACK_WIN_RESULT, DRAW_RESULT, RED_WIN_RESULT, apply_move, encode_fen, iccs_to_indices, iter_unified_games, split_for


def position_value(result: str, position: np.ndarray) -> float:
    if result == DRAW_RESULT:
        return 0.0
    if result not in {RED_WIN_RESULT, BLACK_WIN_RESULT}:
        raise ValueError(f"unsupported game result: {result!r}")
    return 1.0 if bool(position[14, 0, 0]) == (result == RED_WIN_RESULT) else -1.0


def write_shard(dataset_dir: Path, split: str, index: int, buffer: dict[str, list[object]], counts: Counter[str]) -> None:
    if not buffer["positions"]:
        return
    prefix = dataset_dir / f"{split}-{index:03d}"
    np.save(f"{prefix}-positions.npy", np.asarray(buffer["positions"], dtype=np.float32))
    np.save(f"{prefix}-start_indices.npy", np.asarray(buffer["start_indices"], dtype=np.int64))
    np.save(f"{prefix}-end_indices.npy", np.asarray(buffer["end_indices"], dtype=np.int64))
    np.save(f"{prefix}-values.npy", np.asarray(buffer["values"], dtype=np.float32))
    (dataset_dir / f"{split}-{index:03d}.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in buffer["metadata"]), encoding="utf-8"
    )
    counts[f"{split}_samples"] += len(buffer["positions"])
    counts[f"{split}_shards"] += 1
    for values in buffer.values():
        values.clear()


def export_records(records: Iterable[dict[str, object]], output_dir: Path, max_games: int | None, shard_size: int) -> int:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    buffers = {
        split: {key: [] for key in ("positions", "start_indices", "end_indices", "values", "metadata")}
        for split in ("train", "validation", "test")
    }
    shard_numbers = {split: 0 for split in buffers}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    processed = 0
    for record in records:
        if max_games is not None and processed >= max_games:
            break
        processed += 1
        game_id = str(record.get("game_id", ""))
        fen = record.get("fen")
        moves = record.get("moves")
        if not game_id or not isinstance(fen, str) or not isinstance(moves, list):
            counts["skipped_games"] += 1
            continue
        if game_id in seen:
            counts["duplicate_games"] += 1
            continue
        seen.add(game_id)
        try:
            position = encode_fen(fen)
            split = split_for(game_id)
            result = str(record.get("result", "*"))
            for ply, move in enumerate(moves):
                start, end = iccs_to_indices(str(move))
                buffer = buffers[split]
                buffer["positions"].append(position.copy())
                buffer["start_indices"].append(start)
                buffer["end_indices"].append(end)
                buffer["values"].append(position_value(result, position))
                buffer["metadata"].append({**record, "ply": ply})
                position = apply_move(position, start, end)
            counts["valid_games"] += 1
            if len(buffers[split]["positions"]) >= shard_size:
                write_shard(dataset_dir, split, shard_numbers[split], buffers[split], counts)
                shard_numbers[split] += 1
        except (TypeError, ValueError, IndexError) as error:
            counts["skipped_games"] += 1
            counts["encoding_errors"] += 1
            print(f"skip game {game_id}: {error}", file=sys.stderr)
    for split in buffers:
        write_shard(dataset_dir, split, shard_numbers[split], buffers[split], counts)
    manifest = {
        "processed_games": processed,
        "counts": dict(counts),
        "split_rule": "sha256(game_id) modulo 100: 80/10/10",
        "format": "memory-mappable npy shards + jsonl metadata + value labels",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export final Xiangqi training NPY shards.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="normalized JSONL file or shard directory from unify_format.py")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--shard-size", type=int, default=8192)
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if not args.input_jsonl.is_file() and not args.input_jsonl.is_dir():
        parser.error(f"unified JSONL file or shard directory not found: {args.input_jsonl}")
    return export_records(iter_unified_games(args.input_jsonl), args.output_dir, args.max_games, args.shard_size)


if __name__ == "__main__":
    raise SystemExit(main())
