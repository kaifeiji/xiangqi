from __future__ import annotations

import argparse
import time
from pathlib import Path

from annotate_pikafish import PikafishAnnotator

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pikafish CPU binaries on identical value searches.")
    parser.add_argument("--engine", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1 or args.movetime_ms < 1 or args.threads < 1:
        parser.error("samples, movetime-ms and threads must be positive")

    for command in args.engine:
        command = command.expanduser().resolve()
        if not command.is_file():
            parser.error(f"engine not found: {command}")
        print(f"[pikafish] executable={command}", flush=True)
        started = time.perf_counter()
        nodes = 0
        depths = []
        with PikafishAnnotator(
            command,
            depth=None,
            movetime_ms=args.movetime_ms,
            threads=args.threads,
        ) as annotator:
            for _ in range(args.samples):
                annotator._send("ucinewgame")
                result = annotator.annotate(START_FEN, [])
                nodes += int(result["nodes"])
                depths.append(int(result["depth"]))
        elapsed = time.perf_counter() - started
        print({
            "engine": str(command),
            "samples": args.samples,
            "movetime_ms": args.movetime_ms,
            "threads": args.threads,
            "positions_per_second": args.samples / elapsed,
            "average_depth": sum(depths) / len(depths),
            "nodes_per_second": nodes / elapsed,
            "elapsed_seconds": elapsed,
        }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
