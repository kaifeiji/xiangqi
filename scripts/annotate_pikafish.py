from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import msvcrt
import os
import queue
import re
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_utils import (
    apply_move,
    encode_fen,
    iccs_to_indices,
    indices_to_iccs,
    iter_unified_games,
    load_local_env,
    position_to_fen,
    split_for,
    uci_to_iccs,
)

DEFAULT_MULTIPV = 5
SCHEMA_VERSION = 1
SHARD_LOG_NAME = "shard_times.jsonl"
SHARD_LOG_LOCK_NAME = "shard_times.lock"
DEPTH_RE = re.compile(r"\bdepth (\d+)")
NODES_RE = re.compile(r"\bnodes (\d+)")
MULTIPV_RE = re.compile(r"\bmultipv (\d+)")
SCORE_RE = re.compile(r"\bscore (cp|mate) (-?\d+)")
PV_RE = re.compile(r"\bpv (.+)$")


def find_pikafish() -> Path:
    configured = os.environ.get("PIKAFISH_PATH")
    if not configured:
        raise FileNotFoundError("PIKAFISH_PATH is not set")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pikafish executable not found: {path}")
    return path


def find_nnue() -> Path:
    configured = os.environ.get("PIKAFISH_NNUE_PATH")
    if not configured:
        raise FileNotFoundError("PIKAFISH_NNUE_PATH is not set")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pikafish NNUE file not found: {path}")
    return path


def build_teacher(annotation: dict[str, object], *, requested_multipv: int) -> dict[str, object]:
    variations = annotation.get("variations")
    bestmove = annotation.get("bestmove")
    if not isinstance(variations, list) or not isinstance(bestmove, str):
        raise ValueError("Pikafish annotation is missing variations or bestmove")

    candidates: list[dict[str, object]] = []
    for expected_rank, variation in enumerate(variations, start=1):
        if not isinstance(variation, dict):
            raise ValueError("invalid Pikafish variation")
        rank = variation.get("multipv")
        score_kind = variation.get("score_kind")
        score = variation.get("score")
        pv = variation.get("pv")
        if rank != expected_rank or score_kind not in {"cp", "mate"} or not isinstance(score, (int, float)):
            raise ValueError("Pikafish variations must have contiguous ranks and raw scores")
        if not isinstance(pv, list) or not pv or not all(isinstance(move, str) for move in pv):
            raise ValueError("Pikafish variation is missing a principal variation")
        candidate: dict[str, object] = {
            "rank": rank,
            "move": uci_to_iccs(pv[0]),
            "score_kind": score_kind,
            "score": score,
            "depth": int(variation.get("depth", 0)),
            "nodes": int(variation.get("nodes", 0)),
        }
        if requested_multipv > 1:
            candidate["pv"] = [uci_to_iccs(move) for move in pv]
        candidates.append(candidate)

    if not candidates or candidates[0]["move"] != uci_to_iccs(bestmove):
        raise ValueError("Pikafish PV1 does not match bestmove")
    return {
        "score_kind": candidates[0]["score_kind"],
        "score": candidates[0]["score"],
        "bestmove": candidates[0]["move"],
        "requested_multipv": requested_multipv,
        "returned_multipv": len(candidates),
        "candidates": candidates,
    }


class PikafishAnnotator:
    def __init__(
        self,
        command: Path,
        *,
        depth: int | None,
        movetime_ms: int | None,
        nodes: int | None,
        threads: int,
        multipv: int,
        hash_mb: int | None,
    ) -> None:
        if sum(value is not None for value in (depth, movetime_ms, nodes)) != 1:
            raise ValueError("choose exactly one of --depth, --movetime-ms or --nodes")
        self.command = command
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.nodes = nodes
        self.threads = threads
        self.multipv = multipv
        self.hash_mb = hash_mb
        self.process: subprocess.Popen[str] | None = None
        self.lines: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def __enter__(self) -> "PikafishAnnotator":
        nnue_path = find_nnue()
        self.process = subprocess.Popen(
            [str(self.command.expanduser().resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=nnue_path.parent,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None
        threading.Thread(target=self._read, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name EvalFile value {nnue_path}")
        if self.hash_mb is not None:
            self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        if self.multipv > 1:
            self._send(f"setoption name MultiPV value {self.multipv}")
        self._send("isready")
        self._wait_for("readyok")
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Pikafish process is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _wait_for(self, expected: str) -> None:
        while True:
            line = self.lines.get(timeout=30)
            if line is None:
                raise RuntimeError("Pikafish exited unexpectedly")
            if line == expected or line.startswith(expected + " "):
                return

    def annotate(self, fen: str) -> dict[str, object]:
        self._send("position fen " + fen)
        budget = (
            f"depth {self.depth}"
            if self.depth is not None
            else f"movetime {self.movetime_ms}"
            if self.movetime_ms is not None
            else f"nodes {self.nodes}"
        )
        self._send("go " + budget)
        variations: dict[int, dict[str, object]] = {}
        while True:
            line = self.lines.get(timeout=120)
            if line is None:
                raise RuntimeError("Pikafish exited during annotation")
            if line.startswith("info "):
                multipv_match = MULTIPV_RE.search(line)
                multipv = int(multipv_match.group(1)) if multipv_match else 1
                variation = variations.setdefault(multipv, {"multipv": multipv})
                if match := DEPTH_RE.search(line):
                    variation["depth"] = int(match.group(1))
                if match := NODES_RE.search(line):
                    variation["nodes"] = int(match.group(1))
                if match := SCORE_RE.search(line):
                    variation["score_kind"] = match.group(1)
                    variation["score"] = int(match.group(2))
                if match := PV_RE.search(line):
                    variation["pv"] = match.group(1).split()
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                if 1 not in variations or "score" not in variations[1]:
                    raise RuntimeError(f"Pikafish returned no score for {fen}")
                return {"bestmove": bestmove, "variations": [variations[index] for index in sorted(variations)]}


def input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        candidate
        for candidate in sorted(path.glob("*.jsonl"))
        if candidate.name != "duplicates.jsonl" and not candidate.name.endswith(".duplicates.jsonl")
    ]


def output_path_for(source: Path, input_jsonl: Path, output_dir: Path, index: int) -> Path:
    name = "annotated_positions" if input_jsonl.is_file() else source.stem
    return output_dir / f"{name}-{index:03d}.jsonl"


def compressed_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".zst")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def locked_shard_log(output_dir: Path):
    lock_path = output_dir / SHARD_LOG_LOCK_NAME
    with lock_path.open("a+b") as lock_stream:
        if lock_stream.tell() == 0:
            lock_stream.write(b"0")
            lock_stream.flush()
        lock_stream.seek(0)
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield output_dir / SHARD_LOG_NAME
        finally:
            lock_stream.seek(0)
            msvcrt.locking(lock_stream.fileno(), msvcrt.LK_UNLCK, 1)


def read_shard_log(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get("shard"), str):
                    records.append(record)
    return records


def write_shard_log(path: Path, records: list[dict[str, object]]) -> None:
    temporary_path = path.with_suffix(".partial.jsonl")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def shard_has_started_log(output_dir: Path, path: Path) -> bool:
    with locked_shard_log(output_dir) as log_path:
        return any(record.get("shard") == path.name for record in read_shard_log(log_path))


def log_shard_started(output_dir: Path, source: Path, path: Path) -> None:
    with locked_shard_log(output_dir) as log_path:
        records = read_shard_log(log_path)
        if not any(record.get("shard") == path.name for record in records):
            records.append({"source": str(source), "shard": path.name, "started_at": utc_now()})
            write_shard_log(log_path, records)


def log_shard_completed(output_dir: Path, source: Path, path: Path, positions: int) -> None:
    with locked_shard_log(output_dir) as log_path:
        records = read_shard_log(log_path)
        for record in records:
            if record.get("shard") == path.name:
                record["completed_at"] = utc_now()
                record["positions"] = positions
                write_shard_log(log_path, records)
                return
        records.append(
            {
                "source": str(source),
                "shard": path.name,
                "started_at": utc_now(),
                "completed_at": utc_now(),
                "positions": positions,
            }
        )
        write_shard_log(log_path, records)


def output_shards_for(source: Path, input_jsonl: Path, output_dir: Path) -> list[tuple[int, Path]]:
    name = "annotated_positions" if input_jsonl.is_file() else source.stem
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)\.jsonl(?:\.zst)?$")
    shards: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{name}-*.jsonl*"):
        if match := pattern.match(path.name):
            shards.append((int(match.group(1)), path))
    return sorted(shards)


def iter_annotation_records(path: Path):
    if path.suffix == ".zst":
        with path.open("rb") as stream, zstd.ZstdDecompressor().stream_reader(stream) as compressed_stream:
            with io.TextIOWrapper(compressed_stream, encoding="utf-8") as text_stream:
                for line in text_stream:
                    if line.strip():
                        yield json.loads(line)
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def expected_plies_by_game(source: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    for game in iter_unified_games(source):
        game_id = game.get("game_id")
        moves = game.get("moves")
        if isinstance(game_id, str) and isinstance(moves, list):
            expected[game_id] = len(moves)
    return expected


def completed_game_ids(path: Path, expected_plies: dict[str, int]) -> set[str]:
    plies_by_game: dict[str, list[int]] = {}
    for record in iter_annotation_records(path):
        if not isinstance(record, dict):
            continue
        game_id = record.get("game_id")
        ply = record.get("ply")
        if isinstance(game_id, str) and isinstance(ply, int):
            plies_by_game.setdefault(game_id, []).append(ply)
    return {
        game_id
        for game_id, plies in plies_by_game.items()
        if len(plies) == expected_plies.get(game_id)
        and sorted(plies) == list(range(len(plies)))
        and len(set(plies)) == len(plies)
    }


def recover_latest_shard(path: Path, expected_plies: dict[str, int]) -> tuple[set[str], int]:
    records_by_game: dict[str, list[str]] = {}
    plies_by_game: dict[str, list[int]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                game_id = record.get("game_id")
                ply = record.get("ply")
                if isinstance(game_id, str) and isinstance(ply, int):
                    records_by_game.setdefault(game_id, []).append(line)
                    plies_by_game.setdefault(game_id, []).append(ply)
    except json.JSONDecodeError:
        pass
    completed = {
        game_id
        for game_id, plies in plies_by_game.items()
        if len(plies) == expected_plies.get(game_id)
        and sorted(plies) == list(range(len(plies)))
        and len(set(plies)) == len(plies)
    }
    retained = [line for game_id, lines in records_by_game.items() if game_id in completed for line in lines]
    path.write_text("".join(retained), encoding="utf-8")
    return completed, sum(len(records_by_game[game_id]) for game_id in completed)


def compress_shard(path: Path) -> None:
    compressed_path = compressed_path_for(path)
    temporary_path = compressed_path.with_suffix(compressed_path.suffix + ".partial")
    with path.open("rb") as source, temporary_path.open("wb") as destination:
        with zstd.ZstdCompressor(level=3).stream_writer(destination) as compressor:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressor.write(chunk)
    temporary_path.replace(compressed_path)
    path.unlink()


def prepare_output(
    source: Path,
    input_jsonl: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[set[str], int, int, Path]:
    shards = output_shards_for(source, input_jsonl, output_dir)
    if not resume:
        for _, path in shards:
            path.unlink()
        return set(), 0, 0, output_path_for(source, input_jsonl, output_dir, 0)

    expected_plies = expected_plies_by_game(source)
    completed: set[str] = set()
    raw_shards = [(index, path) for index, path in shards if path.suffix == ".jsonl"]
    if len(raw_shards) > 1:
        raise ValueError(f"multiple incomplete output shards for {source.name}")
    if raw_shards and raw_shards[0][0] != shards[-1][0]:
        raise ValueError(f"incomplete output shard is not latest for {source.name}")
    for _, path in shards:
        if path.suffix == ".zst":
            completed.update(completed_game_ids(path, expected_plies))
    if raw_shards:
        index, path = raw_shards[0]
        recovered, positions = recover_latest_shard(path, expected_plies)
        completed.update(recovered)
        return completed, index, positions, path
    next_index = shards[-1][0] + 1 if shards else 0
    return completed, next_index, 0, output_path_for(source, input_jsonl, output_dir, next_index)


def annotate_source(
    source: Path,
    input_jsonl: Path,
    output_dir: Path,
    *,
    max_games: int | None,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    pikafish_threads: int,
    multipv: int,
    hash_mb: int | None,
    shard_size: int,
    resume: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    completed_games, shard_index, shard_positions, records_path = prepare_output(
        source,
        input_jsonl,
        output_dir,
        resume=resume,
    )
    if not records_path.exists() or not shard_has_started_log(output_dir, records_path):
        log_shard_started(output_dir, source, records_path)
    with PikafishAnnotator(
        find_pikafish(),
        depth=depth,
        movetime_ms=movetime_ms,
        nodes=nodes,
        threads=pikafish_threads,
        multipv=multipv,
        hash_mb=hash_mb,
    ) as annotator:
        games = iter(iter_unified_games(source))
        output = records_path.open("a", encoding="utf-8")
        while max_games is None or counts["processed_games"] < max_games:
            try:
                game = next(games)
            except StopIteration:
                break
            game_id = game.get("game_id")
            fen = game.get("fen")
            raw_moves = game.get("moves")
            if isinstance(game_id, str) and game_id in completed_games:
                counts["skipped_games"] += 1
                continue
            counts["processed_games"] += 1
            if not isinstance(game_id, str) or not isinstance(fen, str) or not isinstance(raw_moves, list):
                counts["invalid_games"] += 1
                print("[error] invalid game record", flush=True)
                continue
            try:
                moves = [iccs_to_indices(str(move)) for move in raw_moves]
                position = encode_fen(fen)
                samples: list[dict[str, Any]] = []
                for ply, (start, end) in enumerate(moves):
                    current_fen = position_to_fen(position)
                    teacher = build_teacher(
                        annotator.annotate(current_fen),
                        requested_multipv=multipv,
                    )
                    sample: dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "game_id": game_id,
                        "split": split_for(game_id),
                        "ply": ply,
                        "fen": current_fen,
                        "move": indices_to_iccs(start, end),
                        "teacher": teacher,
                    }
                    samples.append(sample)
                    position = apply_move(position, start, end)
                if not samples:
                    counts["invalid_games"] += 1
                    continue
                if shard_positions >= shard_size:
                    output.close()
                    log_shard_completed(output_dir, source, records_path, shard_positions)
                    compress_shard(records_path)
                    shard_index += 1
                    shard_positions = 0
                    records_path = output_path_for(source, input_jsonl, output_dir, shard_index)
                    log_shard_started(output_dir, source, records_path)
                    output = records_path.open("w", encoding="utf-8")
                for sample in samples:
                    output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                shard_positions += len(samples)
                counts["positions"] += len(samples)
                counts["valid_games"] += 1
                completed_games.add(game_id)
            except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                counts["invalid_games"] += 1
                print(f"[error] source={source.name} game_id={game_id} error={error}", flush=True)
        output.close()
    return counts


def annotate_games(
    input_jsonl: Path,
    output_dir: Path,
    *,
    max_games: int | None,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    pikafish_threads: int,
    multipv: int,
    hash_mb: int | None,
    workers: int,
    shard_size: int,
    resume: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        (output_dir / SHARD_LOG_NAME).unlink(missing_ok=True)
    sources = input_paths(input_jsonl)
    worker_count = min(workers, len(sources))
    common_args = {
        "input_jsonl": input_jsonl,
        "output_dir": output_dir,
        "max_games": max_games,
        "depth": depth,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "pikafish_threads": pikafish_threads,
        "multipv": multipv,
        "hash_mb": hash_mb,
        "shard_size": shard_size,
        "resume": resume,
    }
    counts: Counter[str] = Counter()

    def report_done(source: Path, source_counts: Counter[str]) -> None:
        counts.update(source_counts)

    if worker_count == 1:
        for source in sources:
            report_done(source, annotate_source(source, **common_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(annotate_source, source, **common_args): source for source in sources}
            for future in concurrent.futures.as_completed(futures):
                report_done(futures[future], future.result())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate human-game positions with Pikafish.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="normalized JSONL file or shard directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pikafish_annotations"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Pikafish worker processes for input shard directories")
    parser.add_argument("--shard-size", type=int, default=8192, help="Start a new output shard after a completed game reaches this position count")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest output shard")
    parser.add_argument("--pikafish-threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int)
    parser.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--depth", type=int)
    budget.add_argument("--movetime-ms", type=int)
    budget.add_argument("--nodes", type=int)
    args = parser.parse_args()

    if not args.input_jsonl.is_file() and not args.input_jsonl.is_dir():
        parser.error(f"input not found: {args.input_jsonl}")
    for name in ("max_games", "depth", "movetime_ms", "nodes", "pikafish_threads", "hash_mb", "multipv", "workers", "shard_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    load_local_env()
    return annotate_games(
        args.input_jsonl,
        args.output_dir,
        max_games=args.max_games,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        nodes=args.nodes,
        pikafish_threads=args.pikafish_threads,
        multipv=args.multipv,
        hash_mb=args.hash_mb,
        workers=args.workers,
        shard_size=args.shard_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())