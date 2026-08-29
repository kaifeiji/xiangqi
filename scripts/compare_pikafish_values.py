from __future__ import annotations

import argparse
import math
import queue
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from annotate_pikafish import find_pikafish
from data_utils import load_local_env, uci_to_iccs


SCORE_RE = re.compile(r"\bscore (cp|mate) (-?\d+)")
DEPTH_RE = re.compile(r"\bdepth (\d+)")


class PikafishEvaluator:
    def __init__(self, command: Path, movetime_ms: int) -> None:
        self.command = command
        self.movetime_ms = movetime_ms
        self.process: subprocess.Popen[str] | None = None
        self.lines: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def __enter__(self) -> "PikafishEvaluator":
        self.process = subprocess.Popen(
            [str(self.command)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=self.command.parent, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert self.process.stdout is not None
        threading.Thread(target=self._read, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name ScoreType value Raw")
        self._send("isready")
        self._wait_for("readyok")
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self._send("quit")
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def restart(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None
        self.lines = queue.SimpleQueue()
        self.__enter__()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Pikafish process is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _wait_for(self, expected: str) -> None:
        while True:
            line = self.lines.get(timeout=30)
            if line is None:
                raise RuntimeError("Pikafish exited during initialization")
            if line == expected or line.startswith(expected + " "):
                return

    def evaluate(self, fen: str) -> dict[str, object]:
        self._send("position fen " + fen)
        self._send(f"go movetime {self.movetime_ms}")
        score: tuple[str, int] | None = None
        depth: int | None = None
        timeout_seconds = max(2.0, self.movetime_ms / 1000 * 8)
        while True:
            try:
                line = self.lines.get(timeout=timeout_seconds)
            except queue.Empty as error:
                raise RuntimeError(f"Pikafish timed out after {timeout_seconds:g}s") from error
            if line is None:
                raise RuntimeError("Pikafish exited during evaluation")
            if line.startswith("info "):
                if match := SCORE_RE.search(line):
                    score = (match.group(1), int(match.group(2)))
                if match := DEPTH_RE.search(line):
                    depth = int(match.group(1))
            elif line.startswith("bestmove "):
                if score is None:
                    raise RuntimeError(f"Pikafish returned no score for {fen}")
                return {
                    "bestmove": line.split()[1], "score_kind": score[0],
                    "score": score[1], "depth": depth,
                }


def debug_value(snapshot: dict[str, Any]) -> tuple[dict[str, Any], float] | None:
    for key, value_key in (("mcts_debug", "root_network_value"), ("policy_debug", "network_value")):
        debug = snapshot.get(key)
        if not isinstance(debug, dict):
            continue
        value = debug.get(value_key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return debug, float(value)
    return None


def selected_snapshots(archive: dict[str, Any], turn: int | None) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    snapshots = archive.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("archive snapshots are missing")
    rows = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or (turn is not None and snapshot.get("turn") != turn):
            continue
        result = debug_value(snapshot)
        if result is not None:
            debug, value = result
            rows.append((snapshot, debug, value))
    if turn is not None and not rows:
        raise ValueError(f"no evaluated snapshot found for turn {turn}")
    return rows


def score_text(annotation: dict[str, object]) -> str:
    kind = annotation["score_kind"]
    score = annotation["score"]
    return f"#{score}" if kind == "mate" else f"{score:+} cp"


def table_row(
    snapshot: dict[str, Any], debug: dict[str, Any], network_value: float, annotation: dict[str, object] | None,
    error: str | None = None,
) -> str:
    turn = snapshot.get("turn", "?")
    side = debug.get("searched_side", "?")
    side_name = "红" if side == "w" else "黑" if side == "b" else "?"
    selected = debug.get("selected_move", "--")
    if annotation is None:
        return f"| {turn} | {side_name} | {network_value:+.3f} | error: {error} | -- | {selected} | -- |"
    bestmove = uci_to_iccs(str(annotation["bestmove"]))
    depth = annotation.get("depth") or "?"
    return f"| {turn} | {side_name} | {network_value:+.3f} | {score_text(annotation)} | {depth} | {selected} | {bestmove} |"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare archive root values against Pikafish on identical FENs.")
    parser.add_argument("archive", type=Path, help="Saved game JSON file")
    parser.add_argument("--turn", type=int, help="Evaluate only this post-move snapshot turn")
    parser.add_argument("--movetime-ms", type=int, default=250, help="Pikafish time per position (default: 250)")
    parser.add_argument("--output", type=Path, help="Write the Markdown table to this file")
    args = parser.parse_args()
    if args.movetime_ms <= 0:
        parser.error("--movetime-ms must be positive")

    load_local_env()
    archive = __import__("json").loads(args.archive.read_text(encoding="utf-8-sig"))
    if not isinstance(archive, dict):
        raise ValueError("archive root must be an object")
    rows = selected_snapshots(archive, args.turn)
    lines = [
        f"# Pikafish vs Root Value: {args.archive.name}",
        "",
        "Root value and Pikafish score are both from the current side to move in `searched_fen`.",
        "",
        "| 存档手数 | 走方 | Root value | Pikafish | 深度 | 模型选着 | Pikafish 最佳着 |",
        "| ---: | :---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    with PikafishEvaluator(find_pikafish(), args.movetime_ms) as evaluator:
        for snapshot, debug, network_value in rows:
            fen = debug.get("searched_fen")
            if not isinstance(fen, str):
                raise ValueError(f"turn {snapshot.get('turn')} has no searched_fen")
            try:
                annotation = evaluator.evaluate(fen)
            except (RuntimeError, queue.Empty) as error:
                lines.append(table_row(snapshot, debug, network_value, None, str(error)))
                evaluator.restart()
            else:
                lines.append(table_row(snapshot, debug, network_value, annotation))

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())