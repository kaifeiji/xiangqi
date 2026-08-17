from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from data_encoding import apply_move, encode_fen, indices_to_iccs


CATEGORIES = ("全局", "布局", "残局")
GLOBAL_INPUT_DIRS = (
    "全局",
    "比赛对局",
    "大师专集",
    "近代国手名局",
    "让子局",
    "未分类",
    "实战中局夺子取胜技巧150局",
)
SUPPORTED_SUFFIXES = {".xqf", ".cbr", ".cbl"}
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
PIECE_TO_FEN = {
    0: "1",
    1: "R",
    2: "N",
    3: "B",
    4: "A",
    5: "K",
    6: "C",
    7: "P",
    8: "r",
    9: "n",
    10: "b",
    11: "a",
    12: "k",
    13: "c",
    14: "p",
}
CHAR_TO_FEN = set("RNBAKCP rnbakcp".replace(" ", ""))


@dataclass(frozen=True)
class BookGame:
    category: str
    source: str
    title: str
    result: str
    fen: str
    moves: tuple[tuple[int, int], ...]


def book_game_to_dict(game: BookGame) -> dict[str, Any]:
    return {
        "category": game.category,
        "source": game.source,
        "title": game.title,
        "result": game.result,
        "fen": game.fen,
        "moves": [[start, end] for start, end in game.moves],
    }


def book_game_from_dict(data: dict[str, Any]) -> BookGame:
    return BookGame(
        category=str(data["category"]),
        source=str(data["source"]),
        title=str(data["title"]),
        result=str(data["result"]),
        fen=str(data["fen"]),
        moves=tuple((int(start), int(end)) for start, end in data["moves"]),
    )


def load_checkpoint(path: Path) -> tuple[list[BookGame], set[str]]:
    games: list[BookGame] = []
    completed: set[str] = set()
    if not path.exists():
        return games, completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("status") == "completed":
                completed.add(str(record["source"]))
            elif record.get("status") == "game":
                games.append(book_game_from_dict(record["game"]))
    return games, completed


def append_checkpoint(path: Path, games: list[BookGame], source: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for game in games:
            stream.write(json.dumps({"status": "game", "game": book_game_to_dict(game)}, ensure_ascii=False) + "\n")
        stream.write(json.dumps({"status": "completed", "source": source}, ensure_ascii=False) + "\n")


def category_roots(input_dir: Path) -> dict[str, tuple[Path, ...]]:
    return {
        "全局": tuple(input_dir / name for name in GLOBAL_INPUT_DIRS),
        "布局": (input_dir / "布局",),
        "残局": (input_dir / "残局",),
    }


def iter_source_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path


def iter_all_source_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path


def piece_to_fen(board: Any) -> str:
    native_fen = getattr(board, "to_fen", None)
    if callable(native_fen):
        fields = str(native_fen()).split()
        if len(fields) >= 2:
            return " ".join(fields[:2]) + " - - 0 1"
    rows = getattr(board, "_board", None)
    if rows is None:
        raise ValueError("cchess board has no _board attribute")
    converted: list[list[str | None]] = []
    for row in rows:
        converted_row: list[str | None] = []
        for piece in row:
            if piece is None:
                converted_row.append(None)
            elif isinstance(piece, str):
                converted_row.append(piece if piece in CHAR_TO_FEN else None)
            else:
                converted_row.append(PIECE_TO_FEN.get(int(piece), None))
        converted.append(converted_row)
    if len(converted) != 10 or any(len(row) != 9 for row in converted):
        raise ValueError("unexpected cchess board dimensions")
    fen_rows = []
    for row in converted:
        empty = 0
        tokens: list[str] = []
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
    current = first_move
    while current is not None:
        yield current
        current = getattr(current, "next_move", None)


def move_indices(move: Any) -> tuple[int, int]:
    source = getattr(move, "p_from", None)
    target = getattr(move, "p_to", None)
    if (
        not isinstance(source, (list, tuple))
        or not isinstance(target, (list, tuple))
        or len(source) != 2
        or len(target) != 2
        or any(value is None for value in (*source, *target))
    ):
        raise ValueError("cchess move has no p_from/p_to")
    source_row, source_col = int(source[1]), int(source[0])
    target_row, target_col = int(target[1]), int(target[0])
    return source_row * 9 + source_col, target_row * 9 + target_col


def result_for(info: Any) -> str:
    if not isinstance(info, dict):
        return "*"
    for key in ("result", "Result", "结果"):
        value = str(info.get(key, "*")).strip()
        if value in RESULTS:
            return value
    return "*"


def parse_cchess_game(game: Any, source: str, category: str) -> BookGame:
    first_move = getattr(game, "first_move", None)
    if first_move is None:
        raise ValueError("no first move")
    fen = piece_to_fen(game.init_board)
    parsed_moves: list[tuple[int, int]] = []
    for move in move_chain(first_move):
        try:
            parsed_moves.append(move_indices(move))
        except (TypeError, ValueError) as error:
            if not parsed_moves:
                raise ValueError(f"invalid first move: {error}") from error
            break
    moves = tuple(parsed_moves)
    if not moves:
        raise ValueError("empty main line")
    return BookGame(
        category=category,
        source=source,
        title=Path(source.split("#", 1)[0]).stem,
        result=result_for(getattr(game, "info", {})),
        fen=fen,
        moves=moves,
    )


def parse_source(path: Path, category: str, input_dir: Path) -> list[BookGame]:
    try:
        import cchess
    except ImportError as error:
        raise RuntimeError("missing dependency: install cchess before parsing chess book files") from error
    relative = path.relative_to(input_dir).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".cbl":
        library = cchess.Game.read_from_lib(path)
        games = library.get("games", []) if isinstance(library, dict) else []
        return [
            parse_cchess_game(game, f"{relative}#game-{index:05d}", category)
            for index, game in enumerate(games)
        ]
    if suffix == ".xqf":
        return [parse_cchess_game(cchess.read_from_xqf(path), relative, category)]
    if suffix == ".cbr":
        return [parse_cchess_game(cchess.Game.read_from(path), relative, category)]
    raise ValueError(f"unsupported source suffix: {suffix}")


def game_id(game: BookGame) -> str:
    payload = "|".join((game.category, game.source, game.fen, ",".join(f"{a}-{b}" for a, b in game.moves)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def value_label(result: str, position: np.ndarray) -> float:
    if result in {"1/2-1/2", "*"}:
        return 0.0
    red_to_move = bool(position[14, 0, 0])
    red_won = result == "1-0"
    return 1.0 if red_to_move == red_won else -1.0


def split_for(identifier: str) -> str:
    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def write_category_dataset(games: list[BookGame], output_dir: Path, shard_size: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation", "test")
    metadata_streams = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in splits
    }
    buffers = {
        split: {key: [] for key in ("positions", "starts", "ends", "values")}
        for split in splits
    }
    shard_indices = {split: 0 for split in splits}
    sample_counts = Counter()

    def flush(split: str) -> None:
        split_buffers = buffers[split]
        if not split_buffers["positions"]:
            return
        prefix = output_dir / f"{split}-{shard_indices[split]:05d}"
        np.save(prefix.with_name(prefix.name + "-positions.npy"), np.asarray(split_buffers["positions"], dtype=np.float32))
        np.save(prefix.with_name(prefix.name + "-start_indices.npy"), np.asarray(split_buffers["starts"], dtype=np.int64))
        np.save(prefix.with_name(prefix.name + "-end_indices.npy"), np.asarray(split_buffers["ends"], dtype=np.int64))
        np.save(prefix.with_name(prefix.name + "-values.npy"), np.asarray(split_buffers["values"], dtype=np.float32))
        for values in split_buffers.values():
            values.clear()
        shard_indices[split] += 1

    seen_samples: set[str] = set()
    try:
        for game in games:
            identifier = game_id(game)
            split = split_for(identifier)
            position = encode_fen(game.fen)
            for ply, (start, end) in enumerate(game.moves):
                sample_key = f"{identifier}:{ply}"
                if sample_key in seen_samples:
                    continue
                seen_samples.add(sample_key)
                try:
                    next_position = apply_move(position, start, end)
                except ValueError:
                    break
                metadata_streams[split].write(json.dumps({
                    "game_id": identifier,
                    "category": game.category,
                    "source": game.source,
                    "title": game.title,
                    "result": game.result,
                    "ply": ply,
                    "move": indices_to_iccs(start, end),
                }, ensure_ascii=False) + "\n")
                buffers[split]["positions"].append(position.copy())
                buffers[split]["starts"].append(start)
                buffers[split]["ends"].append(end)
                buffers[split]["values"].append(value_label(game.result, position))
                sample_counts[split] += 1
                position = next_position
                if len(buffers[split]["positions"]) >= shard_size:
                    flush(split)
    finally:
        for stream in metadata_streams.values():
            stream.close()
    for split in splits:
        flush(split)
    return {"samples": dict(sample_counts), "games": len(games), "shards": dict(shard_indices)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare separate Chinese chess book datasets by category.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/chess_book-main"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/chess_book_dataset"))
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--resume", action="store_true", help="resume from per-category parsed-game checkpoints")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("xqf", "cbr", "cbl"),
        default=("xqf", "cbr", "cbl"),
        help="formats to parse; use xqf first for a fast, reliable extraction",
    )
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    selected_suffixes = {f".{suffix}" for suffix in args.formats}

    summary: dict[str, Any] = {"input_dir": str(args.input_dir), "categories": {}, "unsupported": [], "failed": []}
    for category, roots in category_roots(args.input_dir).items():
        parsed: list[BookGame] = []
        checkpoint_path = args.output_dir / category / "parsed_games.checkpoint.jsonl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            parsed, completed_sources = load_checkpoint(checkpoint_path)
            print(f"[{category}] resumed_games={len(parsed)} completed_sources={len(completed_sources)}", flush=True)
        else:
            completed_sources = set()
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        source_files = list(iter_all_source_files(roots))
        for path in source_files:
            if path.suffix.lower() not in selected_suffixes and path.suffix.lower() in SUPPORTED_SUFFIXES:
                summary["unsupported"].append({
                    "category": category,
                    "source": path.relative_to(args.input_dir).as_posix(),
                    "suffix": path.suffix.lower(),
                    "reason": "format not selected",
                })
            elif path.suffix.lower() not in SUPPORTED_SUFFIXES:
                summary["unsupported"].append({
                    "category": category,
                    "source": path.relative_to(args.input_dir).as_posix(),
                    "suffix": path.suffix.lower(),
                })
        for path in iter_source_files(roots):
            if path.suffix.lower() not in selected_suffixes:
                continue
            source = path.relative_to(args.input_dir).as_posix()
            if source in completed_sources:
                print(f"[{category}] skip completed {source}", flush=True)
                continue
            if args.max_games is not None and len(parsed) >= args.max_games:
                break
            try:
                print(f"[{category}] parsing {source}", flush=True)
                remaining = None if args.max_games is None else args.max_games - len(parsed)
                source_games = parse_source(path, category, args.input_dir)
                selected_games = source_games if remaining is None else source_games[:remaining]
                parsed.extend(selected_games)
                append_checkpoint(checkpoint_path, selected_games, source)
                print(f"[{category}] parsed_games={len(parsed)}", flush=True)
            except Exception as error:
                print(f"[{category}] failed {source}: {error}", flush=True)
                summary["failed"].append({"category": category, "source": source, "error": str(error)})
        category_output = args.output_dir / category
        summary["categories"][category] = write_category_dataset(parsed, category_output, args.shard_size)
        print(f"{category}: games={len(parsed)} samples={summary['categories'][category]['samples']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "failed": len(summary["failed"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
