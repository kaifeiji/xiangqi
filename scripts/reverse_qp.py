from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HEADER = b"ccf0"
RECORD_OFFSET = 6
RECORD_WIDTH = 3
QP_PIECES = "KABNRCPkabnrcp"


def read_qp(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(HEADER):
        raise ValueError(f"unexpected header: {data[:4]!r}")
    payload = data[RECORD_OFFSET:]
    piece_count = data[5]
    expected_payload_size = piece_count * RECORD_WIDTH
    payload = data[RECORD_OFFSET : RECORD_OFFSET + expected_payload_size]
    remainder = data[RECORD_OFFSET + expected_payload_size :]
    records: list[list[int]] = []
    terminator: list[int] | None = None
    for index in range(0, len(payload), RECORD_WIDTH):
        record = list(payload[index : index + RECORD_WIDTH])
        records.append(record)
    return {
        "source": str(path),
        "size": len(data),
        "header": data[:4].decode("ascii"),
        "control_byte": data[4],
        "piece_count": piece_count,
        "record_count": len(records),
        "effective_record_count": len(records),
        "terminator": terminator,
        "remainder_bytes": list(remainder),
        "records": records,
        "byte_range": {
            "min": min(payload) if payload else None,
            "max": max(payload) if payload else None,
            "unique": len(set(payload)),
        },
    }


def decode_candidate(report: dict[str, Any]) -> dict[str, Any]:
    board: dict[tuple[int, int], str] = {}
    errors: list[str] = []
    for record in report["records"]:
        code, column, row = record
        if not 0 <= code < len(QP_PIECES):
            errors.append(f"invalid piece code {code}")
            continue
        if not 0 <= column < 9 or not 0 <= row < 10:
            errors.append(f"invalid square {(column, row)}")
            continue
        square = (column, row)
        if square in board:
            errors.append(f"duplicate square {square}")
        board[square] = QP_PIECES[code]
    errors.extend([] if sum(piece == "K" for piece in board.values()) == 1 else ["red king count is not 1"])
    errors.extend([] if sum(piece == "k" for piece in board.values()) == 1 else ["black king count is not 1"])
    rows: list[str] = []
    for row in range(9, -1, -1):
        empty = 0
        rank: list[str] = []
        for column in range(9):
            piece = board.get((column, row))
            if piece is None:
                empty += 1
            else:
                if empty:
                    rank.append(str(empty))
                    empty = 0
                rank.append(piece)
        if empty:
            rank.append(str(empty))
        rows.append("".join(rank))
    return {
        "valid": not errors,
        "errors": errors,
        "fen": "/".join(rows) + " w - - 0 1" if not errors else None,
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    records = []
    sizes = Counter()
    controls = Counter()
    remainders = Counter()
    for path in paths:
        report = read_qp(path)
        records.append(report)
        sizes[report["size"]] += 1
        controls[report["control_byte"]] += 1
        remainders[len(report["remainder_bytes"])] += 1
    return {
        "files": len(records),
        "sizes": dict(sizes),
        "control_bytes": dict(controls),
        "remainder_lengths": dict(remainders),
        "samples": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the proprietary Chinese chess QP format.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decoded-output", type=Path, help="write decoded FEN positions as JSONL")
    args = parser.parse_args()
    paths = sorted(args.input.rglob("*.qp")) + sorted(args.input.rglob("*.QP"))
    paths = paths[: args.limit] if args.limit is not None else paths
    report = summarize(paths)
    decoded = [decode_candidate(sample) for sample in report["samples"]]
    report["decoded_valid"] = sum(item["valid"] for item in decoded)
    report["decoded_invalid"] = len(decoded) - report["decoded_valid"]
    if args.decoded_output:
        with args.decoded_output.open("w", encoding="utf-8") as stream:
            for sample, candidate in zip(report["samples"], decoded):
                stream.write(json.dumps({
                    "source": sample["source"],
                    "piece_count": sample["piece_count"],
                    "fen": candidate["fen"],
                    "valid": candidate["valid"],
                    "errors": candidate["errors"],
                }, ensure_ascii=False) + "\n")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
