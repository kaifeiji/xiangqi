from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from data_utils import load_local_env

SUMMARY_RE = re.compile(r"(?i)bench|nps|nodes|searched|elapsed|time|second")
NPS_PATTERNS = [
    re.compile(r"(?im)\bnps\b[^\d\r\n]*([\d,]+)"),
    re.compile(r"(?im)\bnodes[/ ]second\b[^\d\r\n]*([\d,]+)"),
    re.compile(r"(?im)\bnodes per second\b[^\d\r\n]*([\d,]+)"),
]
NODES_PATTERNS = [
    re.compile(r"(?im)\bnodes searched\b[^\d\r\n]*([\d,]+)"),
    re.compile(r"(?im)\bnodes\b[^\d\r\n]*([\d,]+)"),
]
ELAPSED_PATTERNS = [
    re.compile(r"(?im)\btotal\s*time\s*\(ms\)\b[^\d\r\n]*([\d.]+)"),
    re.compile(r"(?im)\btime\s*\(ms\)\b[^\d\r\n]*([\d.]+)"),
    re.compile(r"(?im)\belapsed\b[^\d\r\n]*([\d.]+)\s*(ms|s|sec|seconds)?"),
]


def configured_pikafish_path() -> Path:
    configured = os.environ.get("PIKAFISH_PATH")
    if not configured:
        raise FileNotFoundError("PIKAFISH_PATH is not set")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pikafish executable not found: {path}")
    return path


def configured_nnue_path() -> Path:
    configured = os.environ.get("PIKAFISH_NNUE_PATH")
    if not configured:
        raise FileNotFoundError("PIKAFISH_NNUE_PATH is not set")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pikafish NNUE file not found: {path}")
    return path


def discover_engines(engine_args: list[Path] | None) -> list[Path]:
    if engine_args:
        engines = [engine.expanduser().resolve() for engine in engine_args]
    else:
        configured = configured_pikafish_path()
        candidates = sorted(configured.parent.glob("pikafish-*.exe"))
        engines = candidates if candidates else [configured]
    missing = [str(engine) for engine in engines if not engine.is_file()]
    if missing:
        raise FileNotFoundError("engine not found: " + ", ".join(missing))
    return engines


def last_int_match(patterns: list[re.Pattern[str]], text: str) -> int | None:
    for pattern in patterns:
        matches = [int(match.group(1).replace(",", "")) for match in pattern.finditer(text)]
        if matches:
            return matches[-1]
    return None


def last_elapsed(text: str) -> str | None:
    for pattern in ELAPSED_PATTERNS:
        match = pattern.search(text)
        if match:
            suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            return f"{match.group(1)} {suffix}".strip()
    return None


def run_builtin_bench(engine: Path, bench_command: str, nnue_path: Path) -> dict[str, Any]:
    command_input = (
        "uci\n"
        f"setoption name EvalFile value {nnue_path}\n"
        "isready\n"
        f"{bench_command}\n"
        "quit\n"
    )
    process = subprocess.run(
        [str(engine)],
        input=command_input,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(nnue_path.parent),
    )
    output = (process.stdout or "") + (process.stderr or "")
    summary_lines = [line for line in output.splitlines() if SUMMARY_RE.search(line)]
    if len(summary_lines) > 12:
        summary_lines = summary_lines[-12:]

    incompatible = False
    if process.returncode in {-1073741795, 3221225501}:
        incompatible = True
    if re.search(r"(?i)illegal instruction|unsupported|not compatible|instruction set|c000001d", output):
        incompatible = True

    return {
        "engine": engine.name,
        "path": str(engine),
        "exitCode": process.returncode,
        "compatible": not incompatible,
        "incompatibleReason": "Unsupported instruction set / illegal instruction" if incompatible else None,
        "nps": last_int_match(NPS_PATTERNS, output),
        "totalNodes": last_int_match(NODES_PATTERNS, output),
        "elapsed": last_elapsed(output),
        "summaryLines": summary_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pikafish CPU binaries using built-in bench.")
    parser.add_argument("--engine", nargs="+", type=Path, help="Engine executables. Defaults to all pikafish-*.exe beside PIKAFISH_PATH")
    parser.add_argument("--bench-command", default="bench", help="UCI command to run benchmark")
    args = parser.parse_args()

    load_local_env()
    nnue_path = configured_nnue_path()
    engines = discover_engines(args.engine)
    results = []
    for engine in engines:
        print(f"[pikafish] executable={engine}", flush=True)
        result = run_builtin_bench(engine, args.bench_command, nnue_path)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    compatible_ranked = sorted(
        [entry for entry in results if entry["compatible"] and entry["nps"] is not None],
        key=lambda entry: int(entry["nps"]),
        reverse=True,
    )
    recommended = compatible_ranked[0]["engine"] if compatible_ranked else next(
        (entry["engine"] for entry in results if entry["compatible"]),
        None,
    )
    report = {
        "enginesTested": [engine.name for engine in engines],
        "rankingByNps": [
            {
                "engine": entry["engine"],
                "nps": entry["nps"],
                "totalNodes": entry["totalNodes"],
                "elapsed": entry["elapsed"],
            }
            for entry in compatible_ranked
        ],
        "recommendedDefaultEngine": recommended,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
