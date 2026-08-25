from __future__ import annotations

import argparse
import hashlib
from itertools import chain
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import apply_move, encode_fen, iccs_to_indices, indices_to_iccs, split_for, validate_fen
from data_utils import EXCLUDED_GAMES
from data_utils import VALID_OR_UNKNOWN_GAME_RESULTS
from data_utils import iter_games

LOGGER = logging.getLogger("unify_format")

RESULTS = VALID_OR_UNKNOWN_GAME_RESULTS
GLOBAL_INPUT_DIRS = ("全局", "比赛对局", "大师专集", "近代国手名局", "让子局", "未分类", "实战中局夺子取胜技巧150局")
PIECE_TO_FEN = {0: "1", 1: "R", 2: "N", 3: "B", 4: "A", 5: "K", 6: "C", 7: "P", 8: "r", 9: "n", 10: "b", 11: "a", 12: "k", 13: "c", 14: "p"}
CHAR_TO_FEN = set("RNBAKCP rnbakcp".replace(" ", ""))


@dataclass(frozen=True)
class BookGame:
    category: str
    source: str
    title: str
    result: str
    fen: str
    moves: tuple[tuple[int, int], ...]


def category_roots(input_dir: Path) -> dict[str, tuple[Path, ...]]:
    return {"全局": tuple(input_dir / name for name in GLOBAL_INPUT_DIRS), "布局": (input_dir / "布局",), "残局": (input_dir / "残局",)}


def iter_source_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_dir():
            yield from sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".xqf")


def piece_to_fen(board: Any) -> str:
    native_fen = getattr(board, "to_fen", None)
    if callable(native_fen):
        fields = str(native_fen()).split()
        if len(fields) >= 2:
            return " ".join(fields[:2]) + " - - 0 1"
    rows = getattr(board, "_board", None)
    if rows is None:
        raise ValueError("cchess board has no _board attribute")
    converted = [[piece if isinstance(piece, str) and piece in CHAR_TO_FEN else PIECE_TO_FEN.get(int(piece)) if piece is not None else None for piece in row] for row in rows]
    if len(converted) != 10 or any(len(row) != 9 for row in converted):
        raise ValueError("unexpected cchess board dimensions")
    fen_rows = []
    for row in converted:
        tokens, empty = [], 0
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty:
                    tokens.append(str(empty))
                    empty = 0
                tokens.append(piece)
        if empty:
            tokens.append(str(empty))
        fen_rows.append("".join(tokens))
    return "/".join(fen_rows) + " w - - 0 1"


def move_chain(first_move: Any) -> Iterator[Any]:
    while first_move is not None:
        yield first_move
        first_move = getattr(first_move, "next_move", None)


def parse_book_game(game: Any, source: str, category: str) -> BookGame:
    first_move = getattr(game, "first_move", None)
    if first_move is None:
        raise ValueError("no first move")
    parsed: list[tuple[int, int]] = []
    for move in move_chain(first_move):
        source_square = getattr(move, "p_from", None)
        target_square = getattr(move, "p_to", None)
        if not isinstance(source_square, (list, tuple)) or not isinstance(target_square, (list, tuple)):
            if parsed:
                break
            raise ValueError("invalid first move")
        parsed.append((int(source_square[1]) * 9 + int(source_square[0]), int(target_square[1]) * 9 + int(target_square[0])))
    if not parsed:
        raise ValueError("empty main line")
    info = getattr(game, "info", {})
    result = "*"
    if isinstance(info, dict):
        for key in ("result", "Result", "结果"):
            value = str(info.get(key, "*")).strip()
            if value in RESULTS:
                result = value
                break
    return BookGame(category, source, Path(source.split("#", 1)[0]).stem, result, piece_to_fen(game.init_board), tuple(parsed))


def parse_source(path: Path, category: str, input_dir: Path) -> list[BookGame]:
    try:
        import cchess
    except ImportError as error:
        raise RuntimeError("missing dependency: install cchess before parsing XQF files") from error
    if path.suffix.lower() != ".xqf":
        raise ValueError(f"unsupported source suffix: {path.suffix}")
    relative = path.relative_to(input_dir).as_posix()
    return [parse_book_game(cchess.read_from_xqf(path), relative, category)]


def canonical_game_id(fen: str, moves: Iterable[str], result: str) -> str:
    payload = "|".join((fen, " ".join(moves), result))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_and_normalize_moves(fen: str, moves: Iterable[str | tuple[int, int]]) -> list[str]:
    position = encode_fen(fen)
    normalized: list[str] = []
    for move in moves:
        if isinstance(move, str):
            start, end = iccs_to_indices(move)
        else:
            start, end = int(move[0]), int(move[1])
        normalized.append(indices_to_iccs(start, end))
        position = apply_move(position, start, end)
    if not normalized:
        raise ValueError("empty main line")
    return normalized


def normalize_record(
    *,
    source_type: str,
    source_file: str,
    fen: str,
    moves: Iterable[str | tuple[int, int]],
    result: str,
    metadata: dict[str, Any],
    source_sha256: str,
    raw_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    if validate_fen(fen):
        raise ValueError("invalid FEN")
    normalized_result = result if result in RESULTS else "*"
    normalized_moves = validate_and_normalize_moves(fen, moves)
    return {
        "source_type": source_type,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "game_id": canonical_game_id(fen, normalized_moves, normalized_result),
        "fen": fen,
        "moves": normalized_moves,
        "result": normalized_result,
        "raw_result": result,
        "raw_tags": raw_tags or {},
        "metadata": metadata,
    }


def iter_pgn_records(paths: Iterable[Path], stats: Counter[str]) -> Iterator[dict[str, Any]]:
    file_hashes: dict[Path, str] = {}
    for path in paths:
        started = time.perf_counter()
        LOGGER.info("Scanning PGN: %s", path)
        file_hashes[path] = _sha256_file(path)
        for game in iter_games(path):
            stats["pgn_games_seen"] += 1
            if stats["pgn_games_seen"] % 1000 == 0:
                LOGGER.info("PGN games scanned: %d (%s)", stats["pgn_games_seen"], path)
            if (path.name, game.game_number) in EXCLUDED_GAMES:
                stats["excluded_games"] += 1
                continue
            if game.parse_errors:
                stats["invalid_games"] += 1
                continue
            fen = game.tags.get("FEN")
            if fen is None or game.tags.get("Format", "").upper() != "ICCS":
                stats["invalid_games"] += 1
                continue
            try:
                yield normalize_record(
                    source_type="pgn",
                    source_file=str(path),
                    fen=fen,
                    moves=game.moves,
                    result=game.tags.get("Result", "*"),
                    metadata={"game_number": game.game_number},
                    source_sha256=file_hashes[path],
                    raw_tags=dict(game.tags),
                )
            except (ValueError, IndexError):
                stats["invalid_games"] += 1
        LOGGER.info("Finished scanning PGN: %s, elapsed=%.1fs", path, time.perf_counter() - started)


def iter_book_records(
    input_dir: Path,
    formats: set[str],
    stats: Counter[str],
) -> Iterator[dict[str, Any]]:
    file_hashes: dict[Path, str] = {}
    for category, roots in category_roots(input_dir).items():
        for path in iter_source_files(roots):
            if path.suffix.lower().lstrip(".") not in formats:
                continue
            try:
                LOGGER.info("Parsing XQF: %s", path)
                file_hashes[path] = _sha256_file(path)
                games = parse_source(path, category, input_dir)
                for index, game in enumerate(games):
                    stats["book_games_seen"] += 1
                    yield normalize_record(
                        source_type="xqp",
                        source_file=game.source,
                        fen=game.fen,
                        moves=game.moves,
                        result=game.result,
                        metadata={
                            "category": game.category,
                            "title": game.title,
                            "source_game_index": index,
                        },
                        source_sha256=file_hashes[path],
                    )
            except (OSError, RuntimeError, ValueError, IndexError):
                stats["invalid_games"] += 1
                LOGGER.warning("Skipping unreadable XQF: %s", path, exc_info=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                game_id = record.get("game_id")
                if isinstance(game_id, str):
                    existing.add(game_id)
    return existing


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_stale_temporary_files(output: Path) -> None:
    for temporary in output.glob("*.tmp"):
        LOGGER.warning("Removing incomplete temporary file: %s", temporary)
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize PGN and xqp games into one per-game JSONL format."
    )
    parser.add_argument("--pgn", nargs="*", type=Path, default=[])
    parser.add_argument("--book-input-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/human_games"))
    parser.add_argument("--max-games", type=int, help="limit normalized games for a smoke test")
    parser.add_argument("--shard-size", type=int, default=8192, help="games per split JSONL shard")
    parser.add_argument("--resume", action="store_true", help="append only games not already in the output")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if not args.pgn and args.book_input_dir is None:
        parser.error("provide --pgn and/or --book-input-dir")
    if args.output.exists() and args.output.is_file():
        parser.error(f"output must be a directory: {args.output}")
    if args.output.exists() and not args.resume and any(args.output.iterdir()):
        parser.error(f"output directory is not empty; use --resume or choose another path: {args.output}")
    for path in args.pgn:
        if not path.is_file():
            parser.error(f"PGN file not found: {path}")
    if args.book_input_dir is not None and not args.book_input_dir.is_dir():
        parser.error(f"book input directory not found: {args.book_input_dir}")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")

    stats: Counter[str] = Counter()
    records: Iterator[dict[str, Any]] = iter_pgn_records(args.pgn, stats)
    if args.book_input_dir is not None:
        records = chain(records, iter_book_records(args.book_input_dir, {"xqf"}, stats))

    existing_ids: set[str] = set()
    if args.resume:
        for existing in args.output.glob("*.jsonl"):
            existing_ids.update(load_existing_ids(existing))
    args.output.mkdir(parents=True, exist_ok=True)
    remove_stale_temporary_files(args.output)
    shard_numbers = {split: 0 for split in ("train", "validation", "test")}
    buffers = {split: [] for split in shard_numbers}
    for split in shard_numbers:
        existing = sorted(args.output.glob(f"{split}-*.jsonl"))
        if existing:
            shard_numbers[split] = max(int(path.stem.split("-")[-1]) for path in existing) + 1
    duplicates_path = args.output / "duplicates.jsonl"
    written = 0
    duplicates = 0
    resumed = 0
    written_ids: set[str] = set()
    with duplicates_path.open("a", encoding="utf-8") as duplicate_stream:
        for record_index, record in enumerate(records):
            if args.max_games is not None and record_index >= args.max_games:
                break
            game_id = str(record["game_id"])
            if game_id in existing_ids:
                resumed += 1
                if resumed == 1 or resumed % 1000 == 0:
                    LOGGER.info("Skipped already completed games: %d", resumed)
                continue
            if game_id in written_ids:
                duplicate_stream.write(json.dumps({"reason": "duplicate_game_id", "record": record}, ensure_ascii=False, separators=(",", ":")) + "\n")
                duplicates += 1
                if duplicates == 1 or duplicates % 1000 == 0:
                    LOGGER.info("Duplicate games found: %d", duplicates)
                continue
            split = split_for(game_id)
            record["split"] = split
            buffers[split].append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            written_ids.add(game_id)
            written += 1
            if len(buffers[split]) >= args.shard_size:
                target = args.output / f"{split}-{shard_numbers[split]:03d}.jsonl"
                atomic_write_text(target, "\n".join(buffers[split]) + "\n")
                LOGGER.info("Wrote shard: %s, games=%d", target, len(buffers[split]))
                buffers[split].clear()
                shard_numbers[split] += 1
    for split, buffer in buffers.items():
        if buffer:
            target = args.output / f"{split}-{shard_numbers[split]:03d}.jsonl"
            atomic_write_text(target, "\n".join(buffer) + "\n")
            LOGGER.info("Wrote final shard: %s, games=%d", target, len(buffer))

    summary = {
        "output_dir": str(args.output),
        "written_games": written,
        "duplicates": duplicates,
        "resumed_games": resumed,
        "shards": {split: shard_numbers[split] for split in shard_numbers},
        "counts": dict(stats),
    }
    summary_path = args.output / "dataset_summary.json"
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("Normalization complete: written=%d resumed=%d duplicates=%d stats=%s", written, resumed, duplicates, dict(stats))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
