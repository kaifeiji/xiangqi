from __future__ import annotations

from backend.game.engine import START_FEN, apply_move, legal_moves, move_to_iccs, parse_fen, position_to_fen


# Curated mainstream first moves from the standard start position.
# These are used instead of all 44 legal first moves to avoid obviously poor openings.
MAINSTREAM_OPENING_MOVES_ICCS = (
    "B2-E2",  # 中炮
    "H2-E2",  # 中炮（另一侧）
    "B2-F2",  # 过宫炮
    "H2-D2",  # 过宫炮（另一侧）
    "B2-D2",  # 士角炮
    "H2-F2",  # 士角炮（另一侧）
    "B0-C2",  # 起马局
    "H0-G2",  # 起马局（另一侧）
    "C0-E2",  # 飞相局
    "G0-E2",  # 飞相局（另一侧）
    "E3-E4",  # 兵五进一
    "C3-C4",  # 仙人指路侧向兵起手
    "G3-G4",  # 仙人指路侧向兵起手（另一侧）
)


def curated_opening_positions() -> tuple[str, ...]:
    start_position = parse_fen(START_FEN)
    legal_by_iccs = {move_to_iccs(move): move for move in legal_moves(start_position)}
    positions: list[str] = []
    seen: set[str] = set()
    for move_iccs in MAINSTREAM_OPENING_MOVES_ICCS:
        move = legal_by_iccs.get(move_iccs)
        if move is None:
            continue
        next_fen = position_to_fen(apply_move(start_position, move))
        if next_fen in seen:
            continue
        seen.add(next_fen)
        positions.append(next_fen)
    if positions:
        return tuple(positions)

    # Fallback: if a curated move becomes invalid due to rule changes, keep feature usable.
    return tuple(position_to_fen(apply_move(start_position, move)) for move in legal_moves(start_position))
