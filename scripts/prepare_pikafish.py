from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "src"))

from annotate_pikafish import SCHEMA_VERSION, iter_annotation_records
from data_utils import PIECE_CHANNELS, current_view_index, current_view_position, iccs_to_move, legal_moves, parse_fen


SPLITS = ("train", "validation", "test")
BOARD_SQUARES = 90
ACTION_SPACE_SIZE = BOARD_SQUARES * BOARD_SQUARES
DEFAULT_MAX_CANDIDATES = 5
PROGRESS_INTERVAL_SECONDS = 5.0
CP_SCORE_KIND = np.uint8(0)
MATE_SCORE_KIND = np.uint8(1)


def action_id(start: int, end: int) -> int:
    return BOARD_SQUARES * start + end


def current_view_action_id(start: int, end: int, red_to_move: bool) -> int:
    if not red_to_move:
        start, end = current_view_index(start), current_view_index(end)
    return action_id(start, end)


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def score_kind_id(kind: object) -> np.uint8:
    if kind == "cp":
        return CP_SCORE_KIND
    if kind == "mate":
        return MATE_SCORE_KIND
    raise ValueError("score_kind_must_be_cp_or_mate")


def position_to_tensor(position: Any) -> np.ndarray:
    board = np.zeros((15, 10, 9), dtype=np.float32)
    for row, cells in enumerate(position.board):
        for column, piece in enumerate(cells):
            if piece is not None:
                board[PIECE_CHANNELS[piece], row, column] = 1.0
    if position.side_to_move == "w":
        board[14, :, :] = 1.0
    return board


def annotation_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    paths = sorted(input_path.glob("*.jsonl.zst"))
    paths.extend(
        path
        for path in sorted(input_path.glob("*.jsonl"))
        if not path.name.endswith(".partial.jsonl") and path.name.split("-", 1)[0] in SPLITS
    )
    return paths


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DistillationShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        max_candidates: int,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.output_dir = output_dir
        self.dataset_dir = output_dir / "dataset"
        self.max_candidates = max_candidates
        self.buffers = {split: self.new_buffer() for split in SPLITS}
        self.shard_numbers = {split: 0 for split in SPLITS}
        self.game_indices: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
        self.games: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        self.counts: Counter[str] = Counter()

    @staticmethod
    def new_buffer() -> dict[str, list[Any]]:
        return {
            "positions": [], "teacher_score_kinds": [], "teacher_scores": [],
            "candidate_action_ids": [],
            "candidate_score_kinds": [], "candidate_scores": [],
            "legal_action_ids": [],
        }

    def add(self, split: str, sample: dict[str, Any]) -> None:
        buffer = self.buffers[split]
        game_id = str(sample["game_id"])
        game_index = self.game_indices[split].get(game_id)
        if game_index is None:
            game_index = len(self.games[split])
            self.game_indices[split][game_id] = game_index
            sample_start = len(buffer["positions"])
            self.games[split].append({
                "game_id": game_id,
                "sample_start": sample_start,
                "sample_end": sample_start,
            })
        elif self.games[split][game_index]["sample_end"] != len(buffer["positions"]):
            raise RuntimeError(f"game samples are not contiguous: {game_id}")
        self.games[split][game_index]["sample_end"] = len(buffer["positions"]) + 1
        for key in buffer:
            buffer[key].append(sample[key])
        self.counts[f"{split}_samples"] += 1
    def flush(self, split: str, prefix_name: str | None = None) -> None:
        buffer = self.buffers[split]
        if not buffer["positions"]:
            return
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.dataset_dir / (prefix_name or f"{split}-{self.shard_numbers[split]:03d}")
        lengths = [len(actions) for actions in buffer["legal_action_ids"]]
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        arrays = {
            "positions": np.asarray(buffer["positions"], dtype=np.float32),
            "teacher_score_kinds": np.asarray(buffer["teacher_score_kinds"], dtype=np.uint8),
            "teacher_scores": np.asarray(buffer["teacher_scores"], dtype=np.float32),
            "candidate_action_ids": np.asarray(buffer["candidate_action_ids"], dtype=np.int16),
            "candidate_score_kinds": np.asarray(buffer["candidate_score_kinds"], dtype=np.uint8),
            "candidate_scores": np.asarray(buffer["candidate_scores"], dtype=np.float32),
            "legal_action_ids": np.concatenate(buffer["legal_action_ids"]).astype(np.int16, copy=False),
            "legal_action_offsets": offsets,
        }
        for name, array in arrays.items():
            write_npy_atomic(Path(f"{prefix}-{name}.npy"), array)
        write_text_atomic(
            Path(f"{prefix}-games.jsonl"),
            "".join(
                json.dumps(game, ensure_ascii=False) + "\n"
                for game in self.games[split]
            ),
        )
        self.counts[f"{split}_shards"] += 1
        self.shard_numbers[split] += 1
        self.buffers[split] = self.new_buffer()

    def finish(self, prefixes: dict[str, str] | None = None) -> None:
        for split in SPLITS:
            self.flush(split, prefixes.get(split) if prefixes else None)


def validate_and_build_sample(record: dict[str, object], *, max_candidates: int) -> tuple[str, dict[str, Any]]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported_schema_version")
    split = record.get("split")
    game_id = record.get("game_id")
    ply = record.get("ply")
    fen = record.get("fen")
    teacher = record.get("teacher")
    if split not in SPLITS or not isinstance(game_id, str) or not game_id or not isinstance(ply, int):
        raise ValueError("invalid_record_identity")
    if not isinstance(fen, str) or not isinstance(teacher, dict):
        raise ValueError("invalid_record_payload")

    game_position = parse_fen(fen)
    red_to_move = game_position.side_to_move == "w"
    legal_absolute = {(move.start, move.end) for move in legal_moves(game_position)}
    legal_action_ids = np.asarray(
        sorted(current_view_action_id(start, end, red_to_move) for start, end in legal_absolute), dtype=np.int16
    )

    score_kind = teacher.get("score_kind")
    score = teacher.get("score")
    teacher_kind_id = score_kind_id(score_kind)
    teacher_score = float(score) if is_finite_number(score) else float("nan")

    candidate_rows: list[tuple[np.uint8, float, int]] = []
    raw_candidates = teacher.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates_missing")
    if len(raw_candidates) > max_candidates:
        raise ValueError("candidate_count_exceeds_max")
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate_invalid")
        candidate_kind = candidate.get("score_kind")
        candidate_score = candidate.get("score")
        if candidate_kind not in {"cp", "mate"} or not is_finite_number(candidate_score):
            raise ValueError("candidate_score_invalid")
        try:
            candidate_move = iccs_to_move(str(candidate.get("move", "")))
        except ValueError:
            raise ValueError("candidate_move_invalid") from None
        absolute_move = (candidate_move.start, candidate_move.end)
        candidate_action = current_view_action_id(*absolute_move, red_to_move)
        candidate_rows.append((score_kind_id(candidate_kind), float(candidate_score), candidate_action))

    candidate_action_ids = np.full(max_candidates, -1, dtype=np.int16)
    candidate_score_kinds = np.full(max_candidates, CP_SCORE_KIND, dtype=np.uint8)
    candidate_scores = np.zeros(max_candidates, dtype=np.float32)
    for index, (candidate_kind, candidate_score, candidate_action) in enumerate(candidate_rows):
        candidate_action_ids[index] = candidate_action
        candidate_score_kinds[index] = candidate_kind
        candidate_scores[index] = candidate_score

    bestmove_audit: str | None = None
    if candidate_rows:
        try:
            bestmove = iccs_to_move(str(teacher.get("bestmove", "")))
            if current_view_action_id(bestmove.start, bestmove.end, red_to_move) != int(candidate_action_ids[0]):
                bestmove_audit = "bestmove_candidate_mismatch"
        except ValueError:
            bestmove_audit = "bestmove_invalid"

    return str(split), {
        "positions": current_view_position(position_to_tensor(game_position)[:14], red_to_move),
        "teacher_score_kinds": teacher_kind_id,
        "teacher_scores": teacher_score,
        "candidate_action_ids": candidate_action_ids,
        "candidate_score_kinds": candidate_score_kinds,
        "candidate_scores": candidate_scores,
        "legal_action_ids": legal_action_ids,
        "game_id": game_id,
        "ply": ply,
        "bestmove_audit": bestmove_audit,
    }


def export_records(
    records: Iterable[dict[str, object]], output_dir: Path, *, max_games: int | None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    shard_prefix: str | None = None,
    write_summary: bool = True,
) -> dict[str, Any]:
    writer = DistillationShardWriter(
        output_dir,
        max_candidates=max_candidates,
    )
    seen_games: set[str] = set()
    game_splits: dict[str, str] = {}
    seen_positions: set[tuple[str, int]] = set()
    started_at = time.perf_counter()
    last_progress_at = started_at
    processed_records = 0

    def log_progress(force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.perf_counter()
        if not force and now - last_progress_at < PROGRESS_INTERVAL_SECONDS:
            return
        elapsed = now - started_at
        exported = sum(writer.counts[f"{split}_samples"] for split in SPLITS)
        rate = exported / elapsed if elapsed > 0 else 0.0
        buffers = " ".join(
            f"{split}_buffer={len(writer.buffers[split]['positions'])}"
            for split in SPLITS
        )
        print(
            f"[prepare-progress] records={processed_records} games={len(seen_games)} "
            f"exported={exported} rate={rate:.1f}/s elapsed_s={elapsed:.1f} {buffers}",
            flush=True,
        )
        last_progress_at = now

    for record in records:
        processed_records += 1
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            writer.counts["records_invalid_identity"] += 1
            continue
        if game_id not in seen_games:
            if max_games is not None and len(seen_games) >= max_games:
                break
            seen_games.add(game_id)
        try:
            split, sample = validate_and_build_sample(record, max_candidates=max_candidates)
            position_key = (game_id, int(sample["ply"]))
            if position_key in seen_positions:
                writer.counts["records_duplicate_position"] += 1
                continue
            seen_positions.add(position_key)
            previous_split = game_splits.setdefault(game_id, split)
            if previous_split != split:
                writer.counts["records_game_split_mismatch"] += 1
                continue
            writer.add(split, sample)
            if sample["bestmove_audit"]:
                writer.counts[f"audit_{sample['bestmove_audit']}"] += 1
        except (TypeError, ValueError, KeyError, IndexError) as error:
            writer.counts[f"records_skipped_{error}"] += 1
        log_progress()
    prefixes = {
        split: shard_prefix if shard_prefix is not None and shard_prefix.startswith(f"{split}-")
        else f"{split}-{shard_prefix}"
        for split in SPLITS
    } if shard_prefix is not None else None
    writer.finish(prefixes)
    log_progress(force=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "format": "current-view Pikafish distillation NPY shards",
        "action_space": {"size": ACTION_SPACE_SIZE, "encoding": "90 * from + to"},
        "score_kinds": {"cp": int(CP_SCORE_KIND), "mate": int(MATE_SCORE_KIND)},
        "max_candidates": max_candidates,
        "counts": dict(writer.counts),
        "games": len(seen_games),
    }
    if write_summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(output_dir / "dataset_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def source_id(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.zst", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def shard_is_complete(output_dir: Path, prefix: str) -> bool:
    dataset_dir = output_dir / "dataset"
    required = (
        "positions", "teacher_score_kinds", "teacher_scores",
        "candidate_action_ids", "candidate_score_kinds", "candidate_scores",
        "legal_action_ids", "legal_action_offsets",
    )
    return all((dataset_dir / f"{prefix}-{name}.npy").is_file() for name in required) and (
        dataset_dir / f"{prefix}-games.jsonl"
    ).is_file()


def existing_shard_counts(output_dir: Path, prefix: str) -> tuple[str, int, int]:
    split = prefix.split("-", 1)[0]
    positions_path = output_dir / "dataset" / f"{prefix}-positions.npy"
    games_path = output_dir / "dataset" / f"{prefix}-games.jsonl"
    games: set[str] = set()
    samples = int(np.load(positions_path, mmap_mode="r", allow_pickle=False).shape[0])
    with games_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            game_id = json.loads(line).get("game_id")
            if isinstance(game_id, str):
                games.add(game_id)
    return split, samples, len(games)


def export_annotation_shards(
    paths: list[Path],
    output_dir: Path,
    *,
    max_candidates: int,
    max_games: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    aggregate: Counter[str] = Counter()
    completed = 0
    skipped = 0
    selected_paths = paths
    for path in selected_paths:
        prefix = source_id(path)
        if shard_is_complete(output_dir, prefix):
            skipped += 1
            completed += 1
            split, samples, games = existing_shard_counts(output_dir, prefix)
            aggregate[f"{split}_samples"] += samples
            aggregate["games"] += games
            print(
                f"[prepare-progress] shards={completed}/{len(selected_paths)} "
                f"current={prefix} status=skipped-complete",
                flush=True,
            )
            continue
        remaining_games = None if max_games is None else max_games - aggregate["games"]
        if remaining_games is not None and remaining_games <= 0:
            break
        shard_started = time.perf_counter()
        summary = export_records(
            iter_annotation_records(path),
            output_dir,
            max_games=remaining_games,
            max_candidates=max_candidates,
            shard_prefix=prefix,
            write_summary=False,
        )
        aggregate.update(summary["counts"])
        aggregate["games"] += int(summary["games"])
        completed += 1
        elapsed = time.perf_counter() - shard_started
        samples = sum(value for key, value in summary["counts"].items() if key.endswith("_samples"))
        print(
            f"[prepare-progress] shards={completed}/{len(selected_paths)} current={prefix} "
            f"status=done records={samples} games={summary['games']} elapsed_s={elapsed:.1f} "
            f"total_elapsed_s={time.perf_counter() - started_at:.1f}",
            flush=True,
        )
    result = {
        "format": "current-view Pikafish distillation NPY shards",
        "max_candidates": max_candidates,
        "counts": dict(aggregate),
        "games": aggregate["games"],
        "annotation_shards": len(selected_paths),
        "completed_shards": completed,
        "skipped_shards": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "dataset_summary.json", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export canonical Pikafish annotations into distillation NPY shards.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="annotate_pikafish.py output file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pikafish-distillation"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if not args.input_jsonl.exists():
        parser.error(f"annotation input not found: {args.input_jsonl}")
    paths = annotation_paths(args.input_jsonl)
    if not paths:
        parser.error("no completed annotation shards found")
    summary = export_annotation_shards(
        paths,
        args.output_dir,
        max_games=args.max_games,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())