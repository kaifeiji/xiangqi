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
import time
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

    bestmove_iccs = uci_to_iccs(bestmove)
    candidates: list[dict[str, object]] = []
    primary_candidate: dict[str, object] | None = None
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
            if expected_rank == 1:
                raise ValueError("Pikafish variation is missing a principal variation")
            continue
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
        if expected_rank == 1:
            primary_candidate = candidate
        candidates.append(candidate)

    if primary_candidate is None:
        raise ValueError("Pikafish variation is missing a principal variation")
    if primary_candidate["move"] != bestmove_iccs:
        raise ValueError("Pikafish PV1 does not match bestmove")
    return {
        "score_kind": primary_candidate["score_kind"],
        "score": primary_candidate["score"],
        "bestmove": primary_candidate["move"],
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
        position_timeout_seconds: float = 5.0,
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
        self.position_timeout_seconds = position_timeout_seconds
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

    def restart(self) -> None:
        self.__exit__(None, None, None)
        self.lines = queue.SimpleQueue()
        self.__enter__()

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
            try:
                line = self.lines.get(timeout=30)
            except queue.Empty as error:
                raise RuntimeError(f"Pikafish timed out waiting for {expected}") from error
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
        deadline = time.monotonic() + self.position_timeout_seconds
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                self.restart()
                raise RuntimeError(
                    f"Pikafish timed out waiting for bestmove after {self.position_timeout_seconds:g}s"
                ) from error
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


def prioritize_sources_for_resume(sources: list[Path], output_dir: Path, *, resume: bool) -> list[Path]:
    if not resume:
        return sources

    def has_incomplete_raw_shard(source: Path) -> bool:
        return any(path.suffix == ".jsonl" for _, path in output_shards_for(source, output_dir))

    # Resume mode: sources with unfinished raw shards should be scheduled first.
    return sorted(sources, key=lambda source: (not has_incomplete_raw_shard(source), source.name))


def output_path_for(source: Path, output_dir: Path, index: int) -> Path:
    return output_dir / f"{source.stem}-{index:03d}.jsonl"


def compressed_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".zst")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_progress_log(**fields: object) -> str:
    parts = ["[progress]"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    return "\t".join(parts)


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


def ensure_shard_started(output_dir: Path, source: Path, path: Path) -> None:
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


def output_shards_for(source: Path, output_dir: Path) -> list[tuple[int, Path]]:
    name = source.stem
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
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[set[str], int, int, Path]:
    shards = output_shards_for(source, output_dir)
    if not resume:
        for _, path in shards:
            path.unlink()
        return set(), 0, 0, output_path_for(source, output_dir, 0)

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
    return completed, next_index, 0, output_path_for(source, output_dir, next_index)


def annotate_source(
    source: Path,
    output_dir: Path,
    *,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    pikafish_threads: int,
    multipv: int,
    hash_mb: int | None,
    shard_size: int,
    resume: bool,
    position_timeout_seconds: float = 5.0,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    progress_log_interval_seconds = 5.0
    completed_games, shard_index, shard_positions, records_path = prepare_output(
        source,
        output_dir,
        resume=resume,
    )
    total_positions = sum(
        int(record.get("positions", 0))
        for record in read_shard_log(output_dir / SHARD_LOG_NAME)
        if record.get("source") == str(source) and isinstance(record.get("positions"), int)
    ) + shard_positions
    print(
        f"[start] source={source.name} resume_completed_games={len(completed_games)} shard_index={shard_index} shard_positions={shard_positions}",
        flush=True,
    )
    progress_started_at = time.monotonic()
    next_progress_log_at = progress_started_at + progress_log_interval_seconds

    def maybe_log_progress(*, force: bool = False) -> None:
        nonlocal next_progress_log_at
        now = time.monotonic()
        if not force and now < next_progress_log_at:
            return
        elapsed_seconds = now - progress_started_at
        print(
            format_progress_log(
                source=source.name,
                processed=len(completed_games) + counts["invalid_games"],
                valid=len(completed_games),
                invalid=counts["invalid_games"],
                shard_index=f"{shard_index:03d}",
                shard_positions=shard_positions,
                total_positions=total_positions,
                elapsed_s=f"{elapsed_seconds:.1f}",
            ),
            flush=True,
        )
        next_progress_log_at = now + progress_log_interval_seconds

    with PikafishAnnotator(
        find_pikafish(),
        depth=depth,
        movetime_ms=movetime_ms,
        nodes=nodes,
        threads=pikafish_threads,
        multipv=multipv,
        hash_mb=hash_mb,
        position_timeout_seconds=position_timeout_seconds,
    ) as annotator:
        dump_json = json.dumps
        games = iter(iter_unified_games(source))
        output = None
        source_game_index = 0
        while True:
            try:
                game = next(games)
            except StopIteration:
                break
            source_game_index += 1
            game_id = game.get("game_id")
            fen = game.get("fen")
            raw_moves = game.get("moves")
            if isinstance(game_id, str) and game_id in completed_games:
                counts["skipped_games"] += 1
                maybe_log_progress()
                continue
            if not isinstance(game_id, str) or not isinstance(fen, str) or not isinstance(raw_moves, list):
                counts["invalid_games"] += 1
                print(
                    f"[fail] source={source.name} game_index={source_game_index} reason=invalid-game-record processed={len(completed_games) + counts['invalid_games']} valid={len(completed_games)} invalid={counts['invalid_games']}",
                    flush=True,
                )
                maybe_log_progress()
                continue
            try:
                moves = [iccs_to_indices(str(move)) for move in raw_moves]
                position = encode_fen(fen)
                game_split = split_for(game_id)
                samples: list[dict[str, Any]] = []
                for ply, (start, end) in enumerate(moves):
                    current_fen = position_to_fen(position)
                    try:
                        teacher = build_teacher(
                            annotator.annotate(current_fen),
                            requested_multipv=multipv,
                        )
                    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                        raise RuntimeError(f"ply={ply} {error}") from error
                    sample: dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "game_id": game_id,
                        "split": game_split,
                        "ply": ply,
                        "fen": current_fen,
                        "move": indices_to_iccs(start, end),
                        "teacher": teacher,
                    }
                    samples.append(sample)
                    position = apply_move(position, start, end)
                if not samples:
                    counts["invalid_games"] += 1
                    print(
                        f"[fail] source={source.name} game_index={source_game_index} reason=no-samples processed={len(completed_games) + counts['invalid_games']} valid={len(completed_games)} invalid={counts['invalid_games']}",
                        flush=True,
                    )
                    maybe_log_progress()
                    continue
                if shard_positions >= shard_size:
                    if output is not None:
                        output.close()
                    print(
                        f"[shard] source={source.name} index={shard_index:03d} positions={shard_positions} action=compress",
                        flush=True,
                    )
                    log_shard_completed(output_dir, source, records_path, shard_positions)
                    compress_shard(records_path)
                    shard_index += 1
                    shard_positions = 0
                    records_path = output_path_for(source, output_dir, shard_index)
                    ensure_shard_started(output_dir, source, records_path)
                    output = records_path.open("w", encoding="utf-8")
                if output is None:
                    ensure_shard_started(output_dir, source, records_path)
                    output = records_path.open("a", encoding="utf-8")
                for sample in samples:
                    output.write(dump_json(sample, ensure_ascii=False) + "\n")
                shard_positions += len(samples)
                total_positions += len(samples)
                counts["positions"] += len(samples)
                counts["valid_games"] += 1
                completed_games.add(game_id)
                maybe_log_progress()
            except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                counts["invalid_games"] += 1
                print(
                    f"[fail] source={source.name} game_index={source_game_index} reason={error} processed={len(completed_games) + counts['invalid_games']} valid={len(completed_games)} invalid={counts['invalid_games']}",
                    flush=True,
                )
                maybe_log_progress()
        if output is not None:
            output.close()
            if shard_positions > 0:
                print(
                    f"[shard] source={source.name} index={shard_index:03d} positions={shard_positions} action=compress-final",
                    flush=True,
                )
                log_shard_completed(output_dir, source, records_path, shard_positions)
                compress_shard(records_path)
        elif shard_positions > 0 and records_path.exists():
            # Resume may recover a non-empty raw shard and then skip all remaining games.
            # In that case no new writes happen, but the recovered tail should still be finalized.
            print(
                f"[shard] source={source.name} index={shard_index:03d} positions={shard_positions} action=compress-recovered",
                flush=True,
            )
            log_shard_completed(output_dir, source, records_path, shard_positions)
            compress_shard(records_path)
        maybe_log_progress(force=True)
    counts["processed_games"] = len(completed_games) + counts["invalid_games"]
    counts["valid_games"] = len(completed_games)
    counts["total_positions"] = total_positions
    print(
        f"[done] source={source.name} processed={counts['processed_games']} valid={counts['valid_games']} invalid={counts['invalid_games']} skipped={counts['skipped_games']} shard_positions={shard_positions} total_positions={total_positions}",
        flush=True,
    )
    return counts


def annotate_games(
    input_jsonl: Path,
    output_dir: Path,
    *,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    pikafish_threads: int,
    multipv: int,
    hash_mb: int | None,
    workers: int,
    shard_size: int,
    resume: bool,
    position_timeout_seconds: float = 5.0,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        (output_dir / SHARD_LOG_NAME).unlink(missing_ok=True)
    sources = prioritize_sources_for_resume(
        input_paths(input_jsonl),
        output_dir,
        resume=resume,
    )
    worker_count = min(workers, len(sources))
    print(
        f"[plan] sources={len(sources)} workers={worker_count} resume={resume} input={input_jsonl}",
        flush=True,
    )
    common_args = {
        "output_dir": output_dir,
        "depth": depth,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "pikafish_threads": pikafish_threads,
        "multipv": multipv,
        "hash_mb": hash_mb,
        "shard_size": shard_size,
        "resume": resume,
        "position_timeout_seconds": position_timeout_seconds,
    }
    counts: Counter[str] = Counter()

    def report_done(source: Path, source_counts: Counter[str]) -> None:
        counts.update(source_counts)
        print(
            f"[worker-done] source={source.name} processed={source_counts['processed_games']} valid={source_counts['valid_games']} invalid={source_counts['invalid_games']} skipped={source_counts['skipped_games']} total_positions={source_counts['total_positions']}",
            flush=True,
        )

    if worker_count == 1:
        for source in sources:
            report_done(source, annotate_source(source, **common_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(annotate_source, source, **common_args): source for source in sources}
            for future in concurrent.futures.as_completed(futures):
                report_done(futures[future], future.result())
    print(
        f"[all-done] processed={counts['processed_games']} valid={counts['valid_games']} invalid={counts['invalid_games']} skipped={counts['skipped_games']} total_positions={counts['total_positions']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate human-game positions with Pikafish.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="normalized JSONL file or shard directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pikafish_annotations"))
    parser.add_argument("--workers", type=int, default=1, help="Pikafish worker processes for input shard directories")
    parser.add_argument("--shard-size", type=int, default=8192, help="Start a new output shard after a completed game reaches this position count")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest output shard")
    parser.add_argument(
        "--position-timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for one Pikafish position (default: 5)",
    )
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
    for name in (
        "depth",
        "movetime_ms",
        "nodes",
        "pikafish_threads",
        "hash_mb",
        "multipv",
        "workers",
        "shard_size",
        "position_timeout",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    load_local_env()
    return annotate_games(
        args.input_jsonl,
        args.output_dir,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        nodes=args.nodes,
        pikafish_threads=args.pikafish_threads,
        multipv=args.multipv,
        hash_mb=args.hash_mb,
        workers=args.workers,
        shard_size=args.shard_size,
        resume=args.resume,
        position_timeout_seconds=args.position_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())