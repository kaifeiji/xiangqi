from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from data_encoding import apply_move, encode_fen, iccs_to_indices


DEFAULT_INPUTS = (
    Path("data/raw/dpxq-99813games.pgns"),
    Path("data/raw/WXF-41743games.pgns"),
)
TAG_RE = re.compile(r'^\s*\[([A-Za-z0-9_]+)\s+"(.*)"\]?\s*$', re.DOTALL)
MOVE_RE = re.compile(r"\b([A-Ia-i][0-9]\s*-\s*[A-Ia-i][0-9])\b")
FEN_PIECES = set("rnbakcpRNBAKCP")
MAX_TAG_BUFFER_BYTES = 1024 * 1024
EXCLUDED_GAMES = {
    ("dpxq-99813games.pgns", 7097),
    ("dpxq-99813games.pgns", 7106),
    ("dpxq-99813games.pgns", 7107),
}


@dataclass
class ParsedGame:
    source_file: str
    game_number: int
    tags: dict[str, str] = field(default_factory=dict)
    movetext: str = ""
    tag_lines_joined: int = 0
    parse_errors: list[str] = field(default_factory=list)

    @property
    def moves(self) -> list[str]:
        return [normalize_move(match.group(1)) for match in MOVE_RE.finditer(self.movetext)]


def normalize_tag_value(value: str) -> str:
    return " ".join(value.split())


def normalize_move(move: str) -> str:
    return re.sub(r"\s+", "", move).upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag_is_complete(text: str) -> bool:
    quote_count = text.count('"')
    stripped = text.rstrip()
    if quote_count < 2:
        return False
    if stripped.endswith("]"):
        return True
    if stripped.endswith('"'):
        return True
    return False


def parse_tag(text: str) -> tuple[str, str] | None:
    match = TAG_RE.match(text)
    if match is None:
        return None
    return match.group(1), normalize_tag_value(match.group(2))


def iter_games(path: Path) -> Iterator[ParsedGame]:
    current: ParsedGame | None = None
    tag_buffer: list[str] = []
    tag_start_line = 0
    game_number = 0

    def flush_tag() -> None:
        nonlocal tag_buffer, tag_start_line
        if not tag_buffer:
            return
        if current is None:
            tag_buffer = []
            return
        parsed = parse_tag("".join(tag_buffer))
        if parsed is None:
            current.parse_errors.append(f"invalid tag near line {tag_start_line}")
        else:
            name, value = parsed
            current.tags[name] = value
            if len(tag_buffer) > 1:
                current.tag_lines_joined += len(tag_buffer) - 1
        tag_buffer = []
        tag_start_line = 0

    def flush_game() -> ParsedGame | None:
        nonlocal current
        if current is None:
            return None
        flush_tag()
        game = current
        current = None
        return game

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.replace("\x00", "")
            stripped = line.strip()

            if tag_buffer:
                if stripped.startswith("[Game ") and current is not None:
                    current.parse_errors.append(
                        f"unclosed tag near line {tag_start_line}; resynchronized at line {line_number}"
                    )
                    tag_buffer = []
                    tag_start_line = 0
                    finished = flush_game()
                    if finished is not None:
                        yield finished
                else:
                    tag_buffer.append(line)
                    if sum(len(part) for part in tag_buffer) > MAX_TAG_BUFFER_BYTES:
                        current.parse_errors.append(
                            f"tag buffer exceeded {MAX_TAG_BUFFER_BYTES} bytes near line {tag_start_line}"
                        )
                        tag_buffer = []
                        tag_start_line = 0
                    elif tag_is_complete("".join(tag_buffer)):
                        flush_tag()
                    continue

            if stripped.startswith("["):
                tag_name_match = re.match(r"\[([A-Za-z0-9_]+)", stripped)
                tag_name = tag_name_match.group(1) if tag_name_match else ""
                if current is not None and tag_name == "Game" and (
                    current.tags or current.movetext
                ):
                    finished = flush_game()
                    if finished is not None:
                        yield finished
                if current is None:
                    game_number += 1
                    current = ParsedGame(str(path), game_number)
                tag_buffer = [line]
                tag_start_line = line_number
                if tag_is_complete(line):
                    flush_tag()
                continue

            if current is None:
                if stripped:
                    game_number += 1
                    current = ParsedGame(str(path), game_number)
                    current.parse_errors.append(f"movetext before tags at line {line_number}")
                else:
                    continue
            current.movetext += line

    if tag_buffer:
        if current is not None:
            current.parse_errors.append(f"unclosed tag near line {tag_start_line}")
        tag_buffer = []
    finished = flush_game()
    if finished is not None:
        yield finished


def validate_fen(fen: str) -> list[str]:
    errors: list[str] = []
    fields = fen.split()
    if len(fields) != 6:
        return ["FEN must contain six fields"]
    ranks = fields[0].split("/")
    if len(ranks) != 10:
        errors.append("FEN board must contain ten ranks")
    for rank_index, rank in enumerate(ranks):
        width = 0
        for character in rank:
            if character.isdigit():
                width += int(character)
            elif character in FEN_PIECES:
                width += 1
            else:
                errors.append(f"invalid FEN piece at rank {rank_index}: {character}")
        if width != 9:
            errors.append(f"FEN rank {rank_index} width is {width}, expected 9")
    if fields[1] not in {"w", "b"}:
        errors.append("FEN side-to-move must be w or b")
    return errors


def game_record(game: ParsedGame, file_hash: str) -> dict[str, object]:
    moves = game.moves
    record = {
        "source_file": game.source_file,
        "game_number": game.game_number,
        "file_sha256": file_hash,
        "tags": game.tags,
        "move_count": len(moves),
        "moves": moves,
        "tag_lines_joined": game.tag_lines_joined,
        "parse_errors": game.parse_errors.copy(),
    }
    fen = game.tags.get("FEN")
    if fen is None:
        record["validation_errors"] = ["missing FEN"]
    else:
        record["validation_errors"] = validate_fen(fen)
    if game.tags.get("Format", "").upper() != "ICCS":
        record["validation_errors"] = list(record["validation_errors"]) + [
            "Format is not ICCS"
        ]
    return record


def scan_paths(paths: Iterable[Path]) -> dict[str, object]:
    summary: dict[str, object] = {
        "files": [],
        "total_games": 0,
        "format_counts": Counter(),
        "result_counts": Counter(),
        "missing_fen": 0,
        "missing_format": 0,
        "tag_lines_joined": 0,
        "unclosed_or_invalid_tags": 0,
        "move_token_counts": Counter(),
    }
    file_records: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            file_records.append({"path": str(path), "error": "file not found"})
            continue
        file_hash = sha256_file(path)
        started = time.perf_counter()
        print(f"scanning {path} ...", flush=True)
        format_counts: Counter[str] = Counter()
        result_counts: Counter[str] = Counter()
        move_counts: Counter[str] = Counter()
        game_count = 0
        missing_fen = 0
        missing_format = 0
        joined = 0
        invalid_tags = 0
        for game in iter_games(path):
            game_count += 1
            if game_count % 1000 == 0:
                elapsed = time.perf_counter() - started
                rate = game_count / elapsed if elapsed else 0.0
                print(
                    f"  {game_count} games ({rate:.0f} games/s)",
                    flush=True,
                )
            format_counts[game.tags.get("Format", "<missing>")] += 1
            result_counts[game.tags.get("Result", "<missing>")] += 1
            move_counts[str(len(game.moves))] += 1
            missing_fen += "FEN" not in game.tags
            missing_format += "Format" not in game.tags
            joined += game.tag_lines_joined
            invalid_tags += sum("tag" in error or "unclosed" in error for error in game.parse_errors)
        summary["total_games"] = int(summary["total_games"]) + game_count
        summary["missing_fen"] = int(summary["missing_fen"]) + missing_fen
        summary["missing_format"] = int(summary["missing_format"]) + missing_format
        summary["tag_lines_joined"] = int(summary["tag_lines_joined"]) + joined
        summary["unclosed_or_invalid_tags"] = int(summary["unclosed_or_invalid_tags"]) + invalid_tags
        summary["format_counts"].update(format_counts)
        summary["result_counts"].update(result_counts)
        summary["move_token_counts"].update(move_counts)
        file_records.append(
            {
                "path": str(path),
                "sha256": file_hash,
                "size_bytes": path.stat().st_size,
                "games": game_count,
                "format_counts": dict(format_counts),
                "result_counts": dict(result_counts),
                "tag_lines_joined": joined,
                "invalid_tag_games": invalid_tags,
            }
        )
        elapsed = time.perf_counter() - started
        print(f"completed {path}: {game_count} games in {elapsed:.1f}s", flush=True)
    summary["files"] = file_records
    for key in ("format_counts", "result_counts", "move_token_counts"):
        summary[key] = dict(summary[key])
    return summary


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_validation(paths: Iterable[Path], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    validated_path = output_dir / "validated_games.jsonl"
    errors_path = output_dir / "data_errors.jsonl"
    summary = Counter()
    with validated_path.open("w", encoding="utf-8") as validated, errors_path.open(
        "w", encoding="utf-8"
    ) as errors:
        for path in paths:
            if not path.exists():
                continue
            file_hash = sha256_file(path)
            for game in iter_games(path):
                record = game_record(game, file_hash)
                all_errors = list(game.parse_errors) + list(record["validation_errors"])
                if all_errors:
                    record["errors"] = all_errors
                    errors.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["error_games"] += 1
                else:
                    validated.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["valid_games"] += 1
    summary["legal_move_validation"] = "not implemented"
    summary["validation_scope"] = "structural only"
    write_json(output_dir / "validation_summary.json", dict(summary))
    return 0 if summary["error_games"] == 0 else 1


def game_split(game: ParsedGame) -> str:
    identity = "|".join(
        [game.tags.get("FEN", ""), " ".join(game.moves), game.tags.get("Result", "")]
    )
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def export_dataset(
    paths: Iterable[Path],
    output_dir: Path,
    max_games: int | None,
    shard_size: int,
) -> int:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, dict[str, list[object]]] = {
        split: {"positions": [], "start_indices": [], "end_indices": [], "metadata": []}
        for split in ("train", "validation", "test")
    }
    shard_numbers = {split: 0 for split in buffers}
    counts = Counter()
    file_hashes: dict[str, str] = {}
    seen_games: set[str] = set()

    def flush(split: str) -> None:
        buffer = buffers[split]
        if not buffer["positions"]:
            return
        shard = shard_numbers[split]
        shard_prefix = dataset_dir / f"{split}-{shard:03d}"
        np.save(f"{shard_prefix}-positions.npy", np.asarray(buffer["positions"], dtype=np.float32))
        np.save(f"{shard_prefix}-start_indices.npy", np.asarray(buffer["start_indices"], dtype=np.int64))
        np.save(f"{shard_prefix}-end_indices.npy", np.asarray(buffer["end_indices"], dtype=np.int64))
        metadata_path = dataset_dir / f"{split}-{shard:03d}.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in buffer["metadata"]),
            encoding="utf-8",
        )
        counts[f"{split}_samples"] += len(buffer["positions"])
        counts[f"{split}_shards"] += 1
        shard_numbers[split] += 1
        for values in buffer.values():
            values.clear()

    processed_games = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        file_hash = sha256_file(path)
        file_hashes[str(path)] = file_hash
        for game in iter_games(path):
            if max_games is not None and processed_games >= max_games:
                break
            processed_games += 1
            if (path.name, game.game_number) in EXCLUDED_GAMES:
                counts["excluded_games"] += 1
                continue
            fen = game.tags.get("FEN")
            if game.parse_errors or fen is None or validate_fen(fen) or game.tags.get("Format", "").upper() != "ICCS":
                counts["skipped_games"] += 1
                continue
            game_id = hashlib.sha256(
                f"{fen}|{' '.join(game.moves)}|{game.tags.get('Result', '*')}".encode("utf-8")
            ).hexdigest()
            if game_id in seen_games:
                counts["duplicate_games"] += 1
                continue
            seen_games.add(game_id)
            split = game_split(game)
            try:
                position = encode_fen(fen)
                for ply, move in enumerate(game.moves):
                    start, end = iccs_to_indices(move)
                    buffer = buffers[split]
                    buffer["positions"].append(position.copy())
                    buffer["start_indices"].append(start)
                    buffer["end_indices"].append(end)
                    buffer["metadata"].append({
                        "game_number": game.game_number,
                        "source_file": game.source_file,
                        "file_sha256": file_hash,
                        "ply": ply,
                        "game_id": game_id,
                        "result": game.tags.get("Result", "*"),
                    })
                    position = apply_move(position, start, end)
                counts["valid_games"] += 1
                if len(buffers[split]["positions"]) >= shard_size:
                    flush(split)
            except (ValueError, IndexError) as error:
                counts["skipped_games"] += 1
                counts["encoding_errors"] += 1
                print(f"skip game {game.game_number}: {error}", file=sys.stderr)
            if processed_games % 1000 == 0:
                print(
                    f"exported {processed_games} games, {sum(value for key, value in counts.items() if key.endswith('_samples'))} samples",
                    flush=True,
                )
        if max_games is not None and processed_games >= max_games:
            break
    for split in buffers:
        flush(split)
    manifest = {
        "processed_games": processed_games,
        "file_sha256": file_hashes,
        "counts": dict(counts),
        "split_rule": "sha256(FEN + moves + Result) modulo 100: 80/10/10",
        "legal_move_validation": "trusted source; not rechecked",
        "format": "memory-mappable npy shards + jsonl metadata",
    }
    write_json(output_dir / "dataset_summary.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def convert_npz_dataset(dataset_dir: Path, overwrite: bool) -> int:
    shard_paths = sorted(
        path
        for split in ("train", "validation", "test")
        for path in dataset_dir.glob(f"{split}-*.npz")
    )
    if not shard_paths:
        raise FileNotFoundError(f"no NPZ shards found under {dataset_dir}")
    converted_samples = 0
    for shard_path in shard_paths:
        prefix = shard_path.with_suffix("")
        outputs = {
            "positions": Path(f"{prefix}-positions.npy"),
            "start_indices": Path(f"{prefix}-start_indices.npy"),
            "end_indices": Path(f"{prefix}-end_indices.npy"),
        }
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"NPY output already exists for {shard_path}; rerun with --overwrite to replace it"
            )
        with np.load(shard_path) as data:
            positions = data["positions"]
            starts = data["start_indices"]
            ends = data["end_indices"]
            if len(positions) != len(starts) or len(positions) != len(ends):
                raise ValueError(f"array lengths do not match in {shard_path}")
            for name, array in (("positions", positions), ("start_indices", starts), ("end_indices", ends)):
                output = outputs[name]
                temporary = output.with_name(f".{output.name}.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, array)
                temporary.replace(output)
            converted_samples += len(positions)
        if converted_samples and converted_samples % 100_000 < len(positions):
            print(f"converted {converted_samples} samples", flush=True)
    summary_path = dataset_dir.parent / "dataset_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["format"] = "memory-mappable npy shards + jsonl metadata"
        summary["converted_from"] = "npz"
        write_json(summary_path, summary)
    print(json.dumps({"converted_shards": len(shard_paths), "converted_samples": converted_samples}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan and structurally validate Xiangqi PGNS files.")
    parser.add_argument("--input", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--convert-npz", type=Path, metavar="DATASET_DIR")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--shard-size", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = (args.scan_only, args.validate, args.export, args.convert_npz is not None)
    if sum(modes) != 1:
        print("choose exactly one of --scan-only, --validate, --export, or --convert-npz")
        return 2
    if args.convert_npz is not None:
        return convert_npz_dataset(args.convert_npz, args.overwrite)
    if args.scan_only:
        summary = scan_paths(args.input)
        write_json(args.output_dir / "data_scan.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        missing_file = any("error" in record for record in summary["files"])
        return 1 if missing_file or summary["total_games"] == 0 else 0
    if args.export:
        return export_dataset(args.input, args.output_dir, args.max_games, args.shard_size)
    return write_validation(args.input, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())