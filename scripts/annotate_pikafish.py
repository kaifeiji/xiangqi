from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_utils import (
    apply_move,
    encode_fen,
    iccs_to_indices,
    iccs_to_uci,
    indices_to_iccs,
    iter_unified_games,
    load_local_env,
    position_to_fen,
    split_for,
    uci_to_iccs,
)

DEFAULT_MULTIPV = 5
SCHEMA_VERSION = 1
TOTAL_PROGRESS_HEARTBEAT_SECONDS = 5.0
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


def build_teacher(
    fen: str,
    annotation: dict[str, object],
    *,
    requested_multipv: int,
    include_pv: bool = False,
) -> dict[str, object]:
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
        candidate = {
            "rank": rank,
            "move": uci_to_iccs(pv[0]),
            "score_kind": score_kind,
            "score": score,
            "depth": int(variation.get("depth", 0)),
            "nodes": int(variation.get("nodes", 0)),
        }
        if include_pv:
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
        threads: int = 1,
        multipv: int = 5,
        hash_mb: int | None = None,
    ) -> None:
        if sum(value is not None for value in (depth, movetime_ms, nodes)) != 1:
            raise ValueError("choose exactly one of --depth, --movetime-ms or --nodes")
        if threads < 1:
            raise ValueError("threads must be positive")
        if multipv < 1:
            raise ValueError("multipv must be positive")
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
        self.command = self.command.expanduser().resolve()
        nnue_path = find_nnue()
        working_dir = nnue_path.parent
        if multiprocessing.current_process().name == "MainProcess":
            print(f"[pikafish] executable={self.command}", flush=True)
            print(f"[pikafish] working_directory={working_dir}", flush=True)
            print(f"[pikafish] nnue={nnue_path} exists={nnue_path.is_file()}", flush=True)
        self.process = subprocess.Popen(
            [str(self.command)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
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
        self.lines.put(None)

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

    def _wait_for(self, expected: str) -> str:
        while True:
            line = self.lines.get(timeout=30)
            if line is None:
                raise RuntimeError("Pikafish exited unexpectedly")
            if line == expected or line.startswith(expected + " "):
                return line

    def annotate(self, fen: str, moves: list[str]) -> dict[str, object]:
        command = "position fen " + fen
        if moves:
            command += " moves " + " ".join(iccs_to_uci(move) for move in moves)
        self._send(command)
        if self.depth is not None:
            budget = f"depth {self.depth}"
        elif self.movetime_ms is not None:
            budget = f"movetime {self.movetime_ms}"
        else:
            budget = f"nodes {self.nodes}"
        self._send("go " + budget)
        variations: dict[int, dict[str, object]] = {}
        while True:
            line = self.lines.get(timeout=120)
            if line is None:
                raise RuntimeError("Pikafish exited during annotation")
            if line.startswith("info "):
                depth_match = DEPTH_RE.search(line)
                nodes_match = NODES_RE.search(line)
                multipv_match = MULTIPV_RE.search(line)
                score_match = SCORE_RE.search(line)
                pv_match = PV_RE.search(line)
                multipv = int(multipv_match.group(1)) if multipv_match else 1
                variation = variations.setdefault(multipv, {"multipv": multipv})
                if depth_match:
                    variation["depth"] = int(depth_match.group(1))
                if nodes_match:
                    variation["nodes"] = int(nodes_match.group(1))
                if score_match:
                    score_kind = score_match.group(1)
                    score = int(score_match.group(2))
                    variation["score_kind"] = score_kind
                    variation["score"] = score
                if pv_match:
                    variation["pv"] = pv_match.group(1).split()
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                primary = variations.get(1)
                if primary is None or "score" not in primary:
                    raise RuntimeError(f"Pikafish returned no score for {fen}")
                ordered = [variations[index] for index in sorted(variations)]
                return {
                    "bestmove": bestmove,
                    "variations": ordered,
                }


def append_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stream.flush()


def append_failure(path: Path, *, game_id: str, source: Path, error: Exception, attempt: int) -> None:
    event = {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "source": str(source),
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp": time.time(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        candidate
        for candidate in sorted(path.glob("*.jsonl"))
        if candidate.name != "duplicates.jsonl" and not candidate.name.endswith(".duplicates.jsonl")
    ]


def output_path_for(source: Path, input_jsonl: Path, output_dir: Path) -> Path:
    if input_jsonl.is_file():
        return output_dir / "annotated_positions.jsonl"
    return output_dir / source.name


def partial_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".partial.jsonl")


def progress_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".progress.json")


def write_progress(
    path: Path,
    counts: Counter[str],
    *,
    complete: bool,
    active_game_id: str | None = None,
    active_ply: int = 0,
    active_positions: int = 0,
) -> None:
    payload = {
        "counts": dict(counts),
        "complete": complete,
        "active_game_id": active_game_id,
        "active_ply": active_ply,
        "active_positions": active_positions,
        "updated_at": time.time(),
    }
    temporary_path = path.with_suffix(".partial.json")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def read_progress(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts") if isinstance(payload, dict) else None
    if not isinstance(counts, dict):
        return Counter()
    progress_counts = Counter({key: value for key, value in counts.items() if isinstance(key, str) and isinstance(value, int)})
    active_positions = payload.get("active_positions") if isinstance(payload, dict) else None
    if isinstance(active_positions, int):
        progress_counts["active_positions"] += active_positions
    return progress_counts


def completed_game_ids(path: Path, expected_plies_by_game: dict[str, int]) -> set[str]:
    if not path.is_file():
        return set()
    plies_by_game: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: invalid annotation record")
            game_id = record.get("game_id")
            ply = record.get("ply")
            if not isinstance(game_id, str) or not isinstance(ply, int):
                raise ValueError(f"{path}:{line_number}: annotation record missing game_id or ply")
            plies_by_game.setdefault(game_id, []).append(ply)
    return {
        game_id
        for game_id, plies in plies_by_game.items()
        if game_id in expected_plies_by_game
        and len(plies) == expected_plies_by_game[game_id]
        and sorted(plies) == list(range(expected_plies_by_game[game_id]))
        and len(set(plies)) == len(plies)
    }


def expected_plies_by_game(source: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    for game in iter_unified_games(source):
        game_id = game.get("game_id")
        moves = game.get("moves")
        if isinstance(game_id, str) and isinstance(moves, list):
            expected[game_id] = len(moves)
    return expected


def discard_incomplete_games(path: Path, completed_games: set[str]) -> None:
    if not path.is_file():
        return
    retained: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("game_id") in completed_games:
                retained.append(line)
    temporary_path = path.with_suffix(".recovered.jsonl")
    with temporary_path.open("w", encoding="utf-8") as stream:
        stream.writelines(retained)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def failure_attempts(path: Path) -> Counter[str]:
    attempts: Counter[str] = Counter()
    if not path.is_file():
        return attempts
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            game_id = event.get("game_id") if isinstance(event, dict) else None
            if not isinstance(game_id, str):
                raise ValueError(f"{path}:{line_number}: failure event missing game_id")
            attempts[game_id] += 1
    return attempts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_shard_metadata(input_jsonl: Path) -> list[dict[str, object]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in input_paths(input_jsonl)
    ]


def input_fingerprint(shards: list[dict[str, object]]) -> str:
    encoded = json.dumps(shards, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def migrate_duplicate_audit_config(existing: dict[str, object]) -> dict[str, object]:
    shards = existing.get("input_shards")
    if not isinstance(shards, list):
        return existing
    retained = [
        shard
        for shard in shards
        if not (
            isinstance(shard, dict)
            and isinstance(shard.get("path"), str)
            and Path(shard["path"]).name == "duplicates.jsonl"
        )
    ]
    if len(retained) == len(shards):
        return existing
    migrated = {**existing, "input_shards": retained, "input_fingerprint": input_fingerprint(retained)}
    return migrated


def migrate_default_runtime_config(existing: dict[str, object]) -> dict[str, object]:
    migrated = dict(existing)
    migrated.setdefault("hash_mb", None)
    migrated.setdefault("progress_every_positions", 16)
    return migrated


def ensure_annotation_config(output_dir: Path, config: dict[str, object]) -> None:
    config = migrate_default_runtime_config(config)
    path = output_dir / "annotation_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            existing = migrate_duplicate_audit_config(existing)
            existing = migrate_default_runtime_config(existing)
        if existing != config:
            raise ValueError(f"annotation configuration does not match {path}; choose a new --output-dir")
        temporary_path = path.with_suffix(".partial.json")
        temporary_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(path)
        return
    temporary_path = path.with_suffix(".partial.json")
    temporary_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def annotate_source(
    source: Path,
    input_root: Path,
    output_dir: Path,
    *,
    max_games: int | None,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    pikafish_threads: int,
    multipv: int,
    resume: bool,
    retry_failed: bool,
    include_pv: bool,
    hash_mb: int | None,
    progress_every_positions: int,
) -> dict[str, int]:
    command = find_pikafish()
    counts: Counter[str] = Counter()
    records_path = output_path_for(source, input_root, output_dir)
    progress_path = progress_path_for(records_path)
    write_progress(progress_path, counts, complete=False)

    with PikafishAnnotator(
        command,
        depth=depth,
        movetime_ms=movetime_ms,
        nodes=nodes,
        threads=pikafish_threads,
        multipv=multipv,
        hash_mb=hash_mb,
    ) as annotator:
        annotation_cache: dict[str, dict[str, object]] = {}
        partial_path = partial_path_for(records_path)
        working_path = records_path if records_path.is_file() else partial_path
        expected_plies = expected_plies_by_game(source)
        done_games = completed_game_ids(working_path, expected_plies) if resume else set()
        if resume and working_path == partial_path:
            discard_incomplete_games(partial_path, done_games)
        failure_path = output_dir / f"{records_path.stem}.failures.jsonl"
        failed_attempts = failure_attempts(failure_path) if resume or retry_failed else Counter()
        retry_games = set(failed_attempts).difference(done_games)

        for game in iter_unified_games(source):
            if max_games is not None and counts["processed_games"] >= max_games:
                break
            counts["processed_games"] += 1
            write_progress(progress_path, counts, complete=False)

            game_id = str(game.get("game_id", ""))
            if not game_id:
                counts["invalid_games"] += 1
                write_progress(progress_path, counts, complete=False)
                continue
            if retry_failed and game_id not in retry_games:
                counts["skipped_games"] += 1
                write_progress(progress_path, counts, complete=False)
                continue
            if not retry_failed and (game_id in done_games or game_id in failed_attempts):
                counts["skipped_games"] += 1
                write_progress(progress_path, counts, complete=False)
                continue

            fen = game.get("fen")
            raw_moves = game.get("moves")
            if not isinstance(fen, str) or not isinstance(raw_moves, list):
                error = ValueError("game is missing fen or moves")
                append_failure(
                    failure_path,
                    game_id=game_id,
                    source=source,
                    error=error,
                    attempt=failed_attempts[game_id] + 1,
                )
                counts["invalid_games"] += 1
                write_progress(progress_path, counts, complete=False)
                continue

            try:
                moves = [iccs_to_indices(str(move)) for move in raw_moves]
                move_texts = [indices_to_iccs(start, end) for start, end in moves]
                position = encode_fen(fen)
                current_moves: list[str] = []
                samples: list[dict[str, Any]] = []
                for ply, (start, end) in enumerate(moves):
                    current_fen = position_to_fen(position)
                    if progress_every_positions > 0 and ply % progress_every_positions == 0:
                        write_progress(
                            progress_path,
                            counts,
                            complete=False,
                            active_game_id=game_id,
                            active_ply=ply,
                            active_positions=ply,
                        )
                    teacher = annotation_cache.get(current_fen)
                    if teacher is None:
                        teacher = build_teacher(
                            current_fen,
                            annotator.annotate(current_fen, []),
                            requested_multipv=multipv,
                            include_pv=include_pv,
                        )
                        annotation_cache[current_fen] = teacher
                    samples.append({
                        "schema_version": SCHEMA_VERSION,
                        "game_id": game_id,
                        "split": split_for(game_id),
                        "ply": ply,
                        "fen": current_fen,
                        "move": move_texts[ply],
                        "teacher": teacher,
                    })
                    position = apply_move(position, start, end)
                    current_moves.append(move_texts[ply])
                write_progress(
                    progress_path,
                    counts,
                    complete=False,
                    active_game_id=game_id,
                    active_ply=len(moves),
                    active_positions=len(moves),
                )

                append_samples(working_path, samples)
                done_games.add(game_id)
                counts["valid_games"] += 1
                counts["positions"] += len(samples)
            except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                append_failure(
                    failure_path,
                    game_id=game_id,
                    source=source,
                    error=error,
                    attempt=failed_attempts[game_id] + 1,
                )
                counts["invalid_games"] += 1
            write_progress(progress_path, counts, complete=False)

        if working_path == partial_path and partial_path.is_file() and (max_games is None or counts["processed_games"] < max_games):
            partial_path.replace(records_path)

    write_progress(progress_path, counts, complete=True)
    return dict(counts)


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
    resume: bool,
    retry_failed: bool,
    workers: int,
    include_pv: bool,
    hash_mb: int | None,
    progress_every_positions: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = input_paths(input_jsonl)
    if workers > 1 and len(sources) == 1:
        raise ValueError("--workers > 1 requires an input shard directory")
    shards = input_shard_metadata(input_jsonl)
    config = {
        "schema_version": SCHEMA_VERSION,
        "input_shards": shards,
        "input_fingerprint": input_fingerprint(shards),
        "depth": depth,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "multipv": multipv,
        "pikafish_threads": pikafish_threads,
        "include_pv": include_pv,
        "hash_mb": hash_mb,
        "progress_every_positions": progress_every_positions,
    }
    ensure_annotation_config(output_dir, config)
    started_at = time.perf_counter()
    next_report_at = started_at + TOTAL_PROGRESS_HEARTBEAT_SECONDS
    worker_count = min(workers, len(sources))
    progress_paths = [progress_path_for(output_path_for(source, input_jsonl, output_dir)) for source in sources]

    def report_progress(*, completed_shards: int) -> None:
        nonlocal next_report_at
        now = time.perf_counter()
        if now < next_report_at:
            return
        counts = sum((read_progress(path) for path in progress_paths), Counter())
        elapsed = now - started_at
        speed = counts["processed_games"] / elapsed if elapsed > 0 else 0.0
        visible_positions = counts["positions"] + counts["active_positions"]
        position_speed = visible_positions / elapsed if elapsed > 0 else 0.0
        print(
            f"[total-progress] completed_shards={completed_shards}/{len(sources)} "
            f"processed_games={counts['processed_games']} valid_games={counts['valid_games']} "
            f"positions={counts['positions']} active_positions={counts['active_positions']} skipped_games={counts['skipped_games']} "
            f"invalid_games={counts['invalid_games']} games_per_second={speed:.3f} "
            f"positions_per_second={position_speed:.3f}",
            flush=True,
        )
        while next_report_at <= now:
            next_report_at += TOTAL_PROGRESS_HEARTBEAT_SECONDS

    common_args = {
        "input_root": input_jsonl,
        "output_dir": output_dir,
        "max_games": max_games,
        "depth": depth,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "pikafish_threads": pikafish_threads,
        "multipv": multipv,
        "resume": resume,
        "retry_failed": retry_failed,
        "include_pv": include_pv,
        "hash_mb": hash_mb,
        "progress_every_positions": progress_every_positions,
    }
    if worker_count == 1:
        counts: Counter[str] = Counter()
        for source in sources:
            counts.update(annotate_source(source, **common_args))
            report_progress(completed_shards=sum(1 for path in sources if output_path_for(path, input_jsonl, output_dir).is_file()))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(annotate_source, source, **common_args) for source in sources]
            pending = set(futures)
            completed_shards = 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=TOTAL_PROGRESS_HEARTBEAT_SECONDS,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                    completed_shards += 1
                report_progress(completed_shards=completed_shards)

    counts = sum((read_progress(path) for path in progress_paths), Counter())
    print(
        f"[total-progress] completed_shards={len(sources)}/{len(sources)} "
        f"processed_games={counts['processed_games']} valid_games={counts['valid_games']} "
        f"positions={counts['positions']} skipped_games={counts['skipped_games']} "
        f"invalid_games={counts['invalid_games']}",
        flush=True,
    )

    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "input_shards": shards,
        "input_fingerprint": config["input_fingerprint"],
        "output_jsonl": [str(output_path_for(path, input_jsonl, output_dir)) for path in sources],
        "multipv": multipv,
        "depth": depth,
        "movetime_ms": movetime_ms,
        "nodes": nodes,
        "include_pv": include_pv,
        "hash_mb": hash_mb,
        "progress_every_positions": progress_every_positions,
        "counts": dict(counts),
        "elapsed_seconds": elapsed_seconds,
        "format": "jsonl per position: fen + move + pikafish teacher labels",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate human-game positions with Pikafish and export JSONL labels."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True, help="normalized JSONL from unify_format.py")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pikafish_annotations"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Pikafish worker processes for input shard directories")
    parser.add_argument("--pikafish-threads", type=int, default=1, help="Threads used by Pikafish")
    parser.add_argument("--hash-mb", type=int, help="Pikafish Hash option in MB per worker")
    parser.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV, help="Number of policy variations to store")
    parser.add_argument("--include-pv", action="store_true", help="Store full ICCS PV lines for every candidate")
    parser.add_argument("--progress-every-positions", type=int, default=16, help="Update worker progress every N plies within an active game; 0 disables in-game updates")
    parser.add_argument("--resume", action="store_true", help="Resume completed games found in annotation JSONL")
    parser.add_argument("--retry-failed", action="store_true", help="Retry games recorded in failure JSONL")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--depth", type=int, help="Pikafish fixed search depth")
    budget.add_argument("--movetime-ms", type=int, help="Pikafish time per position in milliseconds")
    budget.add_argument("--nodes", type=int, help="Pikafish fixed node budget per position")
    args = parser.parse_args()

    if not args.input_jsonl.is_file() and not args.input_jsonl.is_dir():
        parser.error(f"unified JSONL file or shard directory not found: {args.input_jsonl}")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.depth is not None and args.depth < 1:
        parser.error("--depth must be positive")
    if args.movetime_ms is not None and args.movetime_ms < 1:
        parser.error("--movetime-ms must be positive")
    if args.nodes is not None and args.nodes < 1:
        parser.error("--nodes must be positive")
    if args.pikafish_threads < 1:
        parser.error("--pikafish-threads must be positive")
    if args.hash_mb is not None and args.hash_mb < 1:
        parser.error("--hash-mb must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.progress_every_positions < 0:
        parser.error("--progress-every-positions must be non-negative")
    if args.multipv < 1:
        parser.error("--multipv must be positive")

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
        resume=args.resume,
        retry_failed=args.retry_failed,
        workers=args.workers,
        include_pv=args.include_pv,
        hash_mb=args.hash_mb,
        progress_every_positions=args.progress_every_positions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
