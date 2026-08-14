from __future__ import annotations

import re

import numpy as np


BOARD_ROWS = 10
BOARD_COLS = 9
CHANNELS = 15
MOVE_RE = re.compile(r"^([A-Ia-i])([0-9])-([A-Ia-i])([0-9])$")

PIECE_CHANNELS = {
    "K": 0,
    "A": 1,
    "B": 2,
    "N": 3,
    "R": 4,
    "C": 5,
    "P": 6,
    "k": 7,
    "a": 8,
    "b": 9,
    "n": 10,
    "r": 11,
    "c": 12,
    "p": 13,
}


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
    start = square_to_index(match.group(1), int(match.group(2)))
    end = square_to_index(match.group(3), int(match.group(4)))
    return start, end


def indices_to_iccs(start_index: int, end_index: int) -> str:
    return f"{index_to_square(start_index)}-{index_to_square(end_index)}"


def apply_move(position: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    if position.shape != (CHANNELS, BOARD_ROWS, BOARD_COLS):
        raise ValueError(f"expected position shape {(CHANNELS, BOARD_ROWS, BOARD_COLS)}")
    if not 0 <= start_index < BOARD_ROWS * BOARD_COLS:
        raise ValueError(f"invalid start index: {start_index}")
    if not 0 <= end_index < BOARD_ROWS * BOARD_COLS:
        raise ValueError(f"invalid end index: {end_index}")

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
    if len(fields) != 6:
        raise ValueError("FEN must contain six fields")
    ranks = fields[0].split("/")
    if len(ranks) != BOARD_ROWS:
        raise ValueError("FEN board must contain ten ranks")
    if fields[1] not in {"w", "b"}:
        raise ValueError("FEN side-to-move must be w or b")

    board = np.zeros((CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for fen_row, rank in enumerate(ranks):
        iccs_row = BOARD_ROWS - 1 - fen_row
        column = 0
        for piece in rank:
            if piece.isdigit():
                column += int(piece)
                continue
            if piece not in PIECE_CHANNELS or column >= BOARD_COLS:
                raise ValueError(f"invalid FEN board at row {fen_row}")
            board[PIECE_CHANNELS[piece], iccs_row, column] = 1.0
            column += 1
        if column != BOARD_COLS:
            raise ValueError(f"FEN row {fen_row} has width {column}, expected 9")

    if fields[1] == "w":
        board[14, :, :] = 1.0
    return board
