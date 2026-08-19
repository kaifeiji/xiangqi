from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from data_utils import apply_move, encode_fen, iccs_to_indices, indices_to_iccs, iter_unified_games


STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def test_encode_and_replay_move_updates_side_to_move() -> None:
    position = encode_fen(STARTING_FEN)
    start, end = iccs_to_indices("C3-C4")

    assert position.shape == (15, 10, 9)
    assert position.dtype == np.float32
    assert position[14].sum() == 90.0
    assert indices_to_iccs(start, end) == "C3-C4"

    next_position = apply_move(position, start, end)
    start_row, start_column = divmod(start, 9)
    end_row, end_column = divmod(end, 9)
    assert next_position[6, start_row, start_column] == 0.0
    assert next_position[6, end_row, end_column] == 1.0
    assert next_position[14].sum() == 0.0


def test_iter_unified_games_reads_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    record = {"game_id": "game-1", "fen": STARTING_FEN, "moves": ["C3-C4"]}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert list(iter_unified_games(path)) == [record]