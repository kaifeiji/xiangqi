from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

BOARD_ROWS = 10
BOARD_COLS = 9
CHANNELS = 15
MOVE_RE = re.compile(r"^([A-Ia-i])([0-9])-([A-Ia-i])([0-9])$")
PIECE_CHANNELS = {
    "K": 0, "A": 1, "B": 2, "N": 3, "R": 4, "C": 5, "P": 6,
    "k": 7, "a": 8, "b": 9, "n": 10, "r": 11, "c": 12, "p": 13,
}
FEN_PIECES = set(PIECE_CHANNELS)
RED_WIN_RESULT = "1-0"
BLACK_WIN_RESULT = "0-1"
DRAW_RESULT = "1/2-1/2"
VALID_GAME_RESULTS = {RED_WIN_RESULT, BLACK_WIN_RESULT, DRAW_RESULT}
VALID_OR_UNKNOWN_GAME_RESULTS = VALID_GAME_RESULTS | {"*"}
EXCLUDED_GAMES = {
    ("dpxq-99813games.pgns", 7097),
    ("dpxq-99813games.pgns", 7106),
    ("dpxq-99813games.pgns", 7107),
}


def current_view_index(index: int) -> int:
    """Map an ICCS board index after a 180-degree board rotation."""
    return BOARD_ROWS * BOARD_COLS - 1 - index


def current_view_position(position: np.ndarray, red_to_move: bool) -> np.ndarray:
    """Put the side to move at the bottom and keep original red-to-move state."""
    if position.shape != (14, BOARD_ROWS, BOARD_COLS):
        raise ValueError("expected position shape (14, 10, 9)")
    if red_to_move:
        transformed = position.copy()
    else:
        transformed = np.empty((14, BOARD_ROWS, BOARD_COLS), dtype=position.dtype)
        transformed[:7] = np.flip(position[7:14], axis=(1, 2))
        transformed[7:14] = np.flip(position[:7], axis=(1, 2))
    with_side = np.empty((CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=position.dtype)
    with_side[:14] = transformed
    with_side[14, :, :] = 1.0 if red_to_move else 0.0
    return with_side


def side_to_move_sign(position: np.ndarray) -> float:
    """Return +1 for red-to-move and -1 for black-to-move."""
    if position.shape != (CHANNELS, BOARD_ROWS, BOARD_COLS):
        raise ValueError(f"expected position shape {(CHANNELS, BOARD_ROWS, BOARD_COLS)}")
    return 1.0 if bool(position[14, 0, 0]) else -1.0


def load_local_env() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".env.local"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
TAG_RE = re.compile(r'^\s*\[([A-Za-z0-9_]+)\s+"(.*)"\]?\s*$', re.DOTALL)
PGN_MOVE_RE = re.compile(r"\b([A-Ia-i][0-9]\s*-\s*[A-Ia-i][0-9])\b")
MAX_TAG_BUFFER_BYTES = 1024 * 1024


@dataclass
class ParsedGame:
    source_file: str
    game_number: int
    tags: dict[str, str] = field(default_factory=dict)
    movetext: str = ""
    parse_errors: list[str] = field(default_factory=list)

    @property
    def moves(self) -> list[str]:
        return [re.sub(r"\s+", "", match.group(1)).upper() for match in PGN_MOVE_RE.finditer(self.movetext)]


def iter_games(path: Path) -> Iterator[ParsedGame]:
    current: ParsedGame | None = None
    tag_buffer: list[str] = []
    tag_start_line = 0
    game_number = 0

    def finish() -> ParsedGame | None:
        nonlocal current, tag_buffer, tag_start_line
        if current is None:
            return None
        if tag_buffer:
            match = TAG_RE.match("".join(tag_buffer))
            if match:
                current.tags[match.group(1)] = " ".join(match.group(2).split())
            else:
                current.parse_errors.append(f"invalid tag near line {tag_start_line}")
        tag_buffer = []
        tag_start_line = 0
        result = current
        current = None
        return result

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.replace("\x00", "")
            stripped = line.strip()
            if tag_buffer:
                if stripped.startswith("[Game ") and current is not None:
                    current.parse_errors.append(f"unclosed tag near line {tag_start_line}")
                    completed = finish()
                    if completed is not None:
                        yield completed
                else:
                    tag_buffer.append(line)
                    if sum(len(part) for part in tag_buffer) > MAX_TAG_BUFFER_BYTES:
                        current.parse_errors.append(f"tag buffer exceeded {MAX_TAG_BUFFER_BYTES} bytes")
                        tag_buffer = []
                        tag_start_line = 0
                    elif tag_buffer[-1].rstrip().endswith("]") or tag_buffer[-1].rstrip().endswith('"'):
                        match = TAG_RE.match("".join(tag_buffer))
                        if match:
                            current.tags[match.group(1)] = " ".join(match.group(2).split())
                            tag_buffer = []
                            tag_start_line = 0
                    continue
            if stripped.startswith("["):
                match = re.match(r"\[([A-Za-z0-9_]+)", stripped)
                if current is not None and match and match.group(1) == "Game" and (current.tags or current.movetext):
                    completed = finish()
                    if completed is not None:
                        yield completed
                if current is None:
                    game_number += 1
                    current = ParsedGame(str(path), game_number)
                tag_buffer = [line]
                tag_start_line = line_number
                if line.rstrip().endswith("]"):
                    match = TAG_RE.match(line)
                    if match:
                        current.tags[match.group(1)] = " ".join(match.group(2).split())
                        tag_buffer = []
                continue
            if current is None:
                if stripped:
                    game_number += 1
                    current = ParsedGame(str(path), game_number)
                    current.parse_errors.append(f"movetext before tags at line {line_number}")
                continue
            current.movetext += line
    completed = finish()
    if completed is not None:
        yield completed


def square_to_index(column: str, row: int) -> int:
    if column.upper() not in "ABCDEFGHI" or not 0 <= row < BOARD_ROWS:
        raise ValueError(f"invalid ICCS square: {column}{row}")
    return row * BOARD_COLS + (ord(column.upper()) - ord("A"))


def index_to_square(index: int) -> str:
    if not 0 <= index < BOARD_ROWS * BOARD_COLS:
        raise ValueError(f"invalid square index: {index}")
    row, column = divmod(index, BOARD_COLS)
    return f"{chr(ord('A') + column)}{row}"


def iccs_to_indices(move: str) -> tuple[int, int]:
    normalized = re.sub(r"\s+", "", move).upper()
    match = MOVE_RE.match(normalized)
    if match is None:
        raise ValueError(f"invalid ICCS move: {move!r}")
    return (
        square_to_index(match.group(1), int(match.group(2))),
        square_to_index(match.group(3), int(match.group(4))),
    )


def indices_to_iccs(start_index: int, end_index: int) -> str:
    return f"{index_to_square(start_index)}-{index_to_square(end_index)}"


def uci_to_indices(move: str) -> tuple[int, int]:
    normalized = move.strip().lower()
    if len(normalized) != 4:
        raise ValueError(f"invalid UCI move: {move!r}")
    return iccs_to_indices(f"{normalized[:2]}-{normalized[2:]}")


def uci_to_iccs(move: str) -> str:
    return indices_to_iccs(*uci_to_indices(move))


def iccs_to_uci(move: str) -> str:
    start, end = iccs_to_indices(move)
    return indices_to_iccs(start, end).replace("-", "").lower()


def apply_move(position: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    if position.shape != (CHANNELS, BOARD_ROWS, BOARD_COLS):
        raise ValueError(f"expected position shape {(CHANNELS, BOARD_ROWS, BOARD_COLS)}")
    if not 0 <= start_index < BOARD_ROWS * BOARD_COLS or not 0 <= end_index < BOARD_ROWS * BOARD_COLS:
        raise ValueError("invalid move index")
    next_position = position.copy()
    start_row, start_col = divmod(start_index, BOARD_COLS)
    end_row, end_col = divmod(end_index, BOARD_COLS)
    piece_channels = np.flatnonzero(next_position[:14, start_row, start_col])
    if len(piece_channels) != 1:
        raise ValueError(f"expected one piece at start index {start_index}")
    piece_channel = int(piece_channels[0])
    next_position[:14, end_row, end_col] = 0.0
    next_position[piece_channel, start_row, start_col] = 0.0
    next_position[piece_channel, end_row, end_col] = 1.0
    next_position[14, :, :] = 1.0 - next_position[14, :, :]
    return next_position


def encode_fen(fen: str) -> np.ndarray:
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in {"w", "b"}:
        raise ValueError("FEN must contain six fields and a valid side-to-move")
    ranks = fields[0].split("/")
    if len(ranks) != BOARD_ROWS:
        raise ValueError("FEN board must contain ten ranks")
    board = np.zeros((CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for fen_row, rank in enumerate(ranks):
        iccs_row = BOARD_ROWS - 1 - fen_row
        column = 0
        for piece in rank:
            if piece.isdigit():
                column += int(piece)
            elif piece in PIECE_CHANNELS and column < BOARD_COLS:
                board[PIECE_CHANNELS[piece], iccs_row, column] = 1.0
                column += 1
            else:
                raise ValueError(f"invalid FEN board at row {fen_row}")
        if column != BOARD_COLS:
            raise ValueError(f"FEN row {fen_row} has width {column}, expected 9")
    if fields[1] == "w":
        board[14, :, :] = 1.0
    return board


def position_to_fen(position: np.ndarray) -> str:
    if position.shape != (CHANNELS, BOARD_ROWS, BOARD_COLS):
        raise ValueError(f"expected position shape {(CHANNELS, BOARD_ROWS, BOARD_COLS)}")
    ranks: list[str] = []
    channel_to_piece = {channel: piece for piece, channel in PIECE_CHANNELS.items()}
    for fen_row in range(BOARD_ROWS):
        row = BOARD_ROWS - 1 - fen_row
        empty = 0
        tokens: list[str] = []
        for column in range(BOARD_COLS):
            channels = np.flatnonzero(position[:14, row, column])
            if len(channels) == 0:
                empty += 1
                continue
            if len(channels) != 1:
                raise ValueError(f"invalid board encoding at row={row} col={column}")
            if empty:
                tokens.append(str(empty))
                empty = 0
            tokens.append(channel_to_piece[int(channels[0])])
        if empty:
            tokens.append(str(empty))
        ranks.append("".join(tokens))
    side = "w" if bool(position[14, 0, 0]) else "b"
    return "/".join(ranks) + f" {side} - - 0 1"


def validate_fen(fen: str) -> list[str]:
    errors: list[str] = []
    fields = fen.split()
    if len(fields) != 6:
        return ["FEN must contain six fields"]
    ranks = fields[0].split("/")
    if len(ranks) != BOARD_ROWS:
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
        if width != BOARD_COLS:
            errors.append(f"FEN rank {rank_index} width is {width}, expected 9")
    if len(fields) > 1 and fields[1] not in {"w", "b"}:
        errors.append("FEN side-to-move must be w or b")
    return errors


def split_for(game_id: str) -> str:
    bucket = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def iter_unified_games(path: Path) -> Iterator[dict[str, object]]:
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for source_path in paths:
        if source_path.name.endswith(".duplicates.jsonl"):
            continue
        with source_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{source_path}:{line_number}: expected an object")
                yield record