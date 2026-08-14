from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_encoding import encode_fen, iccs_to_indices, indices_to_iccs


def main() -> int:
    sample_path = Path("docs/iccs.sample")
    output_path = Path("artifacts/sample_encoded.json")
    record = next(
        json.loads(line)
        for line in Path("artifacts/smoke_validate/validated_games.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    )
    fen = record["tags"]["FEN"]
    first_move = record["moves"][0]
    board = encode_fen(fen)
    start, end = iccs_to_indices(first_move)
    output = {
        "source": str(sample_path),
        "shape": list(board.shape),
        "dtype": str(board.dtype),
        "nonzero_planes": [int(index) for index in board.any(axis=(1, 2)).nonzero()[0]],
        "first_move": first_move,
        "start_index": start,
        "end_index": end,
        "roundtrip": indices_to_iccs(start, end),
        "side_to_move_plane_sum": float(board[14].sum()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())