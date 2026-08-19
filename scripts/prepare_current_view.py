from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from data_utils import encode_fen, iter_unified_games, split_for


BOARD_COLS = 9
BOARD_ROWS = 10
SIDE_TO_MOVE_CHANNEL = 14
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
ICCS_COLUMNS = "ABCDEFGHI"


def fast_iccs_to_indices(move: str) -> tuple[int, int]:
    if len(move) != 5 or move[2] != "-":
        raise ValueError(f"invalid ICCS move: {move!r}")
    start_column, start_row, end_column, end_row = move[0], move[1], move[3], move[4]
    if start_column not in ICCS_COLUMNS or end_column not in ICCS_COLUMNS:
        raise ValueError(f"invalid ICCS move: {move!r}")
    if not start_row.isdigit() or not end_row.isdigit():
        raise ValueError(f"invalid ICCS move: {move!r}")
    start_row_index = ord(start_row) - ord("0")
    end_row_index = ord(end_row) - ord("0")
    if start_row_index >= BOARD_ROWS or end_row_index >= BOARD_ROWS:
        raise ValueError(f"invalid ICCS move: {move!r}")
    return (
        start_row_index * BOARD_COLS + ord(start_column) - ord("A"),
        end_row_index * BOARD_COLS + ord(end_column) - ord("A"),
    )


def current_view_index(index: int) -> int:
    """Map an ICCS board index after a 180-degree board rotation."""
    return BOARD_ROWS * BOARD_COLS - 1 - index


def current_view_position(position: np.ndarray, red_to_move: bool) -> np.ndarray:
    """Put the side to move at the bottom and return the 14 model channels."""
    if position.shape != (14, BOARD_ROWS, BOARD_COLS):
        raise ValueError("expected position shape (14, 10, 9)")
    if red_to_move:
        return position[:14].copy()

    transformed = np.empty((14, BOARD_ROWS, BOARD_COLS), dtype=position.dtype)
    transformed[:7] = np.flip(position[7:14], axis=(1, 2))
    transformed[7:14] = np.flip(position[:7], axis=(1, 2))
    return transformed


def apply_move_in_place(position: np.ndarray, start_index: int, end_index: int) -> None:
    start_row, start_col = divmod(start_index, BOARD_COLS)
    end_row, end_col = divmod(end_index, BOARD_COLS)
    piece_channels = np.flatnonzero(position[:14, start_row, start_col])
    if len(piece_channels) != 1:
        raise ValueError(f"expected one piece at start index {start_index}")
    piece_channel = int(piece_channels[0])
    position[:14, end_row, end_col] = 0.0
    position[piece_channel, start_row, start_col] = 0.0
    position[piece_channel, end_row, end_col] = 1.0
def write_shard_without_metadata(
    dataset_dir: Path,
    split: str,
    index: int,
    buffer: dict[str, list[object]],
    counts: Counter[str],
    progress_path: Path,
) -> None:
    if not buffer["positions"]:
        return
    prefix = dataset_dir / f"{split}-{index:03d}"
    np.save(f"{prefix}-positions.npy", np.asarray(buffer["positions"], dtype=np.float32))
    np.save(f"{prefix}-start_indices.npy", np.asarray(buffer["start_indices"], dtype=np.int64))
    np.save(f"{prefix}-end_indices.npy", np.asarray(buffer["end_indices"], dtype=np.int64))
    np.save(f"{prefix}-values.npy", np.asarray(buffer["values"], dtype=np.float32))
    with progress_path.open("a", encoding="utf-8") as stream:
        stream.writelines(f"{game_id}\n" for game_id in buffer["game_ids"])
    counts[f"{split}_samples"] += len(buffer["positions"])
    counts[f"{split}_shards"] += 1
    for values in buffer.values():
        values.clear()


def remove_incomplete_shards(dataset_dir: Path) -> None:
    suffixes = ("positions", "start_indices", "end_indices", "values")
    prefixes = {
        path.name[: -len("-positions.npy")]
        for path in dataset_dir.glob("*-positions.npy")
    }
    for prefix in prefixes:
        paths = [dataset_dir / f"{prefix}-{suffix}.npy" for suffix in suffixes]
        if not all(path.is_file() for path in paths):
            for path in paths:
                if path.exists():
                    path.unlink()


def export_records(records, output_dir: Path, max_games: int | None, shard_size: int) -> int:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    remove_incomplete_shards(dataset_dir)
    for metadata_path in dataset_dir.glob("*.jsonl"):
        metadata_path.unlink()
    progress_path = output_dir / "processed_game_ids.txt"
    completed_game_ids = (
        set(progress_path.read_text(encoding="utf-8").splitlines())
        if progress_path.is_file()
        else set()
    )
    buffers = {
        split: {key: [] for key in ("positions", "start_indices", "end_indices", "values", "game_ids")}
        for split in ("train", "validation", "test")
    }
    shard_numbers = {
        split: max(
            [
                int(path.name.split("-")[1])
                for path in dataset_dir.glob(f"{split}-*-positions.npy")
                if path.name.split("-")[1].isdigit()
            ],
            default=-1,
        )
        + 1
        for split in buffers
    }
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    processed = 0

    for record in records:
        if max_games is not None and processed >= max_games:
            break
        processed += 1
        if processed % 100 == 0:
            split_samples = {
                split: len(buffer["positions"])
                for split, buffer in buffers.items()
            }
            print(
                f"processed_games={processed} "
                f"valid_games={counts['valid_games']} "
                f"skipped_games={counts['skipped_games']} "
                f"duplicate_games={counts['duplicate_games']} "
                f"buffer_samples={split_samples}",
                flush=True,
            )
        game_id = str(record.get("game_id", ""))
        fen = record.get("fen")
        moves = record.get("moves")
        if not game_id or not isinstance(fen, str) or not isinstance(moves, list):
            counts["skipped_games"] = counts.get("skipped_games", 0) + 1
            continue
        if game_id in seen:
            counts["duplicate_games"] = counts.get("duplicate_games", 0) + 1
            continue
        if game_id in completed_game_ids:
            counts["resumed_games"] += 1
            seen.add(game_id)
            continue
        seen.add(game_id)
        result = str(record.get("result", "*"))
        if result not in VALID_RESULTS:
            counts["skipped_games"] += 1
            counts["encoding_errors"] += 1
            print(f"skip game {game_id}: unsupported game result: {result!r}", file=sys.stderr)
            continue
        try:
            encoded_position = encode_fen(fen)
            red_to_move = bool(encoded_position[SIDE_TO_MOVE_CHANNEL, 0, 0])
            position = encoded_position[:14].copy()
            split = split_for(game_id)
            for ply, move in enumerate(moves):
                start, end = fast_iccs_to_indices(str(move))
                buffer = buffers[split]
                buffer["positions"].append(current_view_position(position, red_to_move))
                if red_to_move:
                    buffer["start_indices"].append(start)
                    buffer["end_indices"].append(end)
                else:
                    buffer["start_indices"].append(current_view_index(start))
                    buffer["end_indices"].append(current_view_index(end))
                buffer["values"].append(
                    0.0
                    if result == "1/2-1/2"
                    else 1.0 if red_to_move == (result == "1-0") else -1.0
                )
                apply_move_in_place(position, start, end)
                red_to_move = not red_to_move
            buffers[split]["game_ids"].append(game_id)
            counts["valid_games"] = counts.get("valid_games", 0) + 1
            if len(buffers[split]["positions"]) >= shard_size:
                write_shard_without_metadata(
                    dataset_dir, split, shard_numbers[split], buffers[split], counts, progress_path
                )
                shard_numbers[split] += 1
        except (TypeError, ValueError, IndexError) as error:
            counts["skipped_games"] = counts.get("skipped_games", 0) + 1
            counts["encoding_errors"] = counts.get("encoding_errors", 0) + 1
            print(f"skip game {game_id}: {error}", file=sys.stderr)

    for split in buffers:
        write_shard_without_metadata(
            dataset_dir, split, shard_numbers[split], buffers[split], counts, progress_path
        )
    manifest = {
        "processed_games": processed,
        "counts": counts,
        "split_rule": "sha256(game_id) modulo 100: 80/10/10",
        "format": "current-side-view memory-mappable npy shards + value labels",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export current-side-view Xiangqi training NPY shards.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/current_view"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--shard-size", type=int, default=8192)
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if not args.input_jsonl.is_file() and not args.input_jsonl.is_dir():
        parser.error(f"unified JSONL file or shard directory not found: {args.input_jsonl}")
    return export_records(iter_unified_games(args.input_jsonl), args.output_dir, args.max_games, args.shard_size)


if __name__ == "__main__":
    raise SystemExit(main())