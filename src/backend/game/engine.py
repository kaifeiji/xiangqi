from __future__ import annotations

from dataclasses import dataclass

BOARD_ROWS = 10
BOARD_COLS = 9

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

_RED_PALACE_ROWS = {0, 1, 2}
_BLACK_PALACE_ROWS = {7, 8, 9}
_PALACE_COLS = {3, 4, 5}


@dataclass(frozen=True)
class Move:
    start: int
    end: int


@dataclass(frozen=True)
class Position:
    board: tuple[tuple[str | None, ...], ...]
    side_to_move: str


def index_to_coord(index: int) -> tuple[int, int]:
    if not 0 <= index < BOARD_ROWS * BOARD_COLS:
        raise ValueError(f"invalid square index: {index}")
    return divmod(index, BOARD_COLS)


def coord_to_index(row: int, col: int) -> int:
    if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
        raise ValueError(f"invalid coordinate: {(row, col)}")
    return row * BOARD_COLS + col


def index_to_square(index: int) -> str:
    row, col = index_to_coord(index)
    return f"{chr(ord('A') + col)}{row}"


def square_to_index(square: str) -> int:
    normalized = square.strip().upper()
    if len(normalized) != 2 or normalized[0] not in "ABCDEFGHI" or normalized[1] not in "0123456789":
        raise ValueError(f"invalid ICCS square: {square!r}")
    return coord_to_index(int(normalized[1]), ord(normalized[0]) - ord("A"))


def iccs_to_move(text: str) -> Move:
    compact = text.strip().upper().replace(" ", "")
    if len(compact) != 5 or compact[2] != "-":
        raise ValueError(f"invalid ICCS move: {text!r}")
    return Move(square_to_index(compact[:2]), square_to_index(compact[3:]))


def move_to_iccs(move: Move) -> str:
    return f"{index_to_square(move.start)}-{index_to_square(move.end)}"


def parse_fen(fen: str) -> Position:
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must contain six fields")
    board_field, side = fields[0], fields[1]
    if side not in {"w", "b"}:
        raise ValueError("FEN side-to-move must be w or b")
    ranks = board_field.split("/")
    if len(ranks) != BOARD_ROWS:
        raise ValueError("FEN board must contain ten ranks")

    rows: list[tuple[str | None, ...]] = []
    for fen_rank in reversed(ranks):
        cols: list[str | None] = []
        for token in fen_rank:
            if token.isdigit():
                cols.extend([None] * int(token))
            else:
                cols.append(token)
        if len(cols) != BOARD_COLS:
            raise ValueError("FEN row width must be 9")
        rows.append(tuple(cols))
    return Position(tuple(rows), side)


def _to_mutable_board(position: Position) -> list[list[str | None]]:
    return [list(row) for row in position.board]


def _freeze_board(board: list[list[str | None]]) -> tuple[tuple[str | None, ...], ...]:
    return tuple(tuple(row) for row in board)


def _inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS


def _is_red(piece: str) -> bool:
    return piece.isupper()


def _own_piece(piece: str, side: str) -> bool:
    return (_is_red(piece) and side == "w") or (not _is_red(piece) and side == "b")


def _enemy_piece(piece: str, side: str) -> bool:
    return not _own_piece(piece, side)


def _in_palace(row: int, col: int, side: str) -> bool:
    if col not in _PALACE_COLS:
        return False
    return row in (_RED_PALACE_ROWS if side == "w" else _BLACK_PALACE_ROWS)


def _piece_at(position: Position, row: int, col: int) -> str | None:
    if not _inside(row, col):
        return None
    return position.board[row][col]


def _king_square(position: Position, side: str) -> tuple[int, int]:
    target = "K" if side == "w" else "k"
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if position.board[row][col] == target:
                return row, col
    raise ValueError(f"missing king for side {side}")


def _piece_moves(position: Position, row: int, col: int, piece: str) -> list[Move]:
    side = "w" if _is_red(piece) else "b"
    kind = piece.upper()
    moves: list[Move] = []
    start = coord_to_index(row, col)

    def add_if_target(r: int, c: int) -> None:
        if not _inside(r, c):
            return
        target = _piece_at(position, r, c)
        if target is None or _enemy_piece(target, side):
            moves.append(Move(start, coord_to_index(r, c)))

    if kind == "K":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if _inside(nr, nc) and _in_palace(nr, nc, side):
                add_if_target(nr, nc)

        step = 1 if side == "w" else -1
        nr = row + step
        while _inside(nr, col):
            target = _piece_at(position, nr, col)
            if target is not None:
                if target.upper() == "K" and _enemy_piece(target, side):
                    moves.append(Move(start, coord_to_index(nr, col)))
                break
            nr += step
        return moves

    if kind == "A":
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = row + dr, col + dc
            if _inside(nr, nc) and _in_palace(nr, nc, side):
                add_if_target(nr, nc)
        return moves

    if kind == "B":
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            nr, nc = row + dr, col + dc
            eye_r, eye_c = row + dr // 2, col + dc // 2
            if not _inside(nr, nc):
                continue
            if _piece_at(position, eye_r, eye_c) is not None:
                continue
            if side == "w" and nr > 4:
                continue
            if side == "b" and nr < 5:
                continue
            add_if_target(nr, nc)
        return moves

    if kind == "N":
        candidates = (
            (2, 1, 1, 0),
            (2, -1, 1, 0),
            (-2, 1, -1, 0),
            (-2, -1, -1, 0),
            (1, 2, 0, 1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (-1, -2, 0, -1),
        )
        for dr, dc, leg_r, leg_c in candidates:
            if _piece_at(position, row + leg_r, col + leg_c) is not None:
                continue
            add_if_target(row + dr, col + dc)
        return moves

    if kind == "R":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            while _inside(nr, nc):
                target = _piece_at(position, nr, nc)
                if target is None:
                    moves.append(Move(start, coord_to_index(nr, nc)))
                else:
                    if _enemy_piece(target, side):
                        moves.append(Move(start, coord_to_index(nr, nc)))
                    break
                nr += dr
                nc += dc
        return moves

    if kind == "C":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            jumped = False
            while _inside(nr, nc):
                target = _piece_at(position, nr, nc)
                if not jumped:
                    if target is None:
                        moves.append(Move(start, coord_to_index(nr, nc)))
                    else:
                        jumped = True
                else:
                    if target is not None:
                        if _enemy_piece(target, side):
                            moves.append(Move(start, coord_to_index(nr, nc)))
                        break
                nr += dr
                nc += dc
        return moves

    if kind == "P":
        forward = 1 if side == "w" else -1
        add_if_target(row + forward, col)
        crossed = row >= 5 if side == "w" else row <= 4
        if crossed:
            add_if_target(row, col - 1)
            add_if_target(row, col + 1)
        return moves

    raise ValueError(f"unknown piece: {piece}")


def apply_move(position: Position, move: Move) -> Position:
    start_row, start_col = index_to_coord(move.start)
    end_row, end_col = index_to_coord(move.end)
    piece = _piece_at(position, start_row, start_col)
    if piece is None:
        raise ValueError(f"empty start square: {move_to_iccs(move)}")
    if not _own_piece(piece, position.side_to_move):
        raise ValueError(f"cannot move enemy piece: {move_to_iccs(move)}")

    board = _to_mutable_board(position)
    board[end_row][end_col] = piece
    board[start_row][start_col] = None
    return Position(_freeze_board(board), "b" if position.side_to_move == "w" else "w")


def is_in_check(position: Position, side: str) -> bool:
    king_row, king_col = _king_square(position, side)
    enemy_side = "b" if side == "w" else "w"
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = _piece_at(position, row, col)
            if piece is None or not _own_piece(piece, enemy_side):
                continue
            for move in _piece_moves(position, row, col, piece):
                end_row, end_col = index_to_coord(move.end)
                if (end_row, end_col) == (king_row, king_col):
                    return True
    return False


def legal_moves(position: Position) -> list[Move]:
    candidates: list[Move] = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = _piece_at(position, row, col)
            if piece is None or not _own_piece(piece, position.side_to_move):
                continue
            candidates.extend(_piece_moves(position, row, col, piece))

    legal: list[Move] = []
    for move in candidates:
        next_position = apply_move(position, move)
        if not is_in_check(next_position, position.side_to_move):
            legal.append(move)
    return legal


def king_exists(position: Position, side: str) -> bool:
    target = "K" if side == "w" else "k"
    return any(target in row for row in position.board)


def render_board(position: Position) -> str:
    rows: list[str] = []
    for row in range(BOARD_ROWS - 1, -1, -1):
        cells = [position.board[row][col] or "." for col in range(BOARD_COLS)]
        rows.append(f"{row} " + " ".join(cells))
    rows.append("  A B C D E F G H I")
    rows.append(f"side: {'red(w)' if position.side_to_move == 'w' else 'black(b)'}")
    return "\n".join(rows)
