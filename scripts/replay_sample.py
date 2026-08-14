from __future__ import annotations

import json
import sys
from pathlib import Path

from data_encoding import apply_move, encode_fen, iccs_to_indices


def main() -> int:
    record = next(
        json.loads(line)
        for line in Path("artifacts/smoke_validate/validated_games.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    )
    position = encode_fen(record["tags"]["FEN"])
    initial_side_sum = float(position[14].sum())
    for ply, move in enumerate(record["moves"][:10], start=1):
        start, end = iccs_to_indices(move)
        position = apply_move(position, start, end)
        expected_side_sum = 0.0 if ply % 2 else 90.0
        if float(position[14].sum()) != expected_side_sum:
            raise AssertionError(f"side-to-move mismatch after ply {ply}")
    result = {
        "plies_replayed": 10,
        "initial_side_plane_sum": initial_side_sum,
        "final_side_plane_sum": float(position[14].sum()),
        "shape": list(position.shape),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())