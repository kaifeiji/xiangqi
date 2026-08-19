from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path, WindowsPath
from typing import Protocol

import numpy as np
import torch
from torch import nn

from backend.inference.move_scoring import apply_legal_move_mask, joint_move_logits, legal_move_mask
from backend.models import ResNet

from .engine import (
    BOARD_COLS,
    BOARD_ROWS,
    Move,
    Position,
    apply_move,
    iccs_to_move,
    king_exists,
    legal_moves,
    move_to_iccs,
    position_to_fen,
)
from .mcts import MCTS

_PIECE_CHANNELS: dict[str, int] = {
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


class Player(Protocol):
    name: str

    def choose_move(self, position: Position) -> Move: ...


def pikafish_command() -> str | None:
    configured_path = os.environ.get("PIKAFISH_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        return str(path) if path.is_file() else None
    path_command = shutil.which("pikafish") or shutil.which("pikafish.exe")
    if path_command:
        return path_command

    workspace_dir = Path(__file__).resolve().parents[4]
    for install_dir in sorted(workspace_dir.glob("Pikafish.*"), reverse=True):
        candidate = install_dir / "Windows" / "pikafish-avx2.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def position_to_tensor(position: Position) -> torch.Tensor:
    board = np.zeros((15, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = position.board[row][col]
            if piece is not None:
                board[_PIECE_CHANNELS[piece], row, col] = 1.0
    if position.side_to_move == "w":
        board[14, :, :] = 1.0
    return torch.from_numpy(board)


def current_view_index(index: int) -> int:
    return BOARD_ROWS * BOARD_COLS - 1 - index


def current_view_tensor(position: Position, board: torch.Tensor) -> torch.Tensor:
    if position.side_to_move == "w":
        return board
    transformed = torch.empty_like(board)
    transformed[:7] = torch.flip(board[7:14], dims=[1, 2])
    transformed[7:14] = torch.flip(board[:7], dims=[1, 2])
    transformed[14] = 1.0
    return transformed


@dataclass
class HumanPlayer:
    name: str

    def choose_move(self, position: Position) -> Move:
        legal = legal_moves(position)
        legal_set = {(move.start, move.end) for move in legal}
        while True:
            command = input(f"{self.name} 输入走法(ICCS, 例如 A0-A1，输入 quit 退出): ").strip()
            if command.lower() in {"quit", "exit", "resign"}:
                raise KeyboardInterrupt(f"{self.name} resigned")
            try:
                move = iccs_to_move(command)
            except ValueError as error:
                print(f"无效输入: {error}")
                continue
            if (move.start, move.end) not in legal_set:
                print("该走法不合法，请重试。")
                continue
            return move


@dataclass
class ModelPlayer:
    name: str
    model: nn.Module
    device: torch.device
    sampling_temperature: float | None = None
    sampling_top_k: int = 5
    current_view: bool = False
    input_channels: int = 15
    value_head: bool = False
    mcts_time_seconds: float = 0.0
    opening_plies: int = 2
    opening_temperature: float = 0.5

    @classmethod
    def from_checkpoint(
        cls,
        *,
        name: str,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        sampling_temperature: float | None = None,
        sampling_top_k: int = 5,
        current_view: bool | None = None,
        mcts_time_seconds: float = 0.0,
    ) -> "ModelPlayer":
        if sampling_temperature is not None and sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if sampling_top_k < 1:
            raise ValueError("sampling_top_k must be positive")
        if mcts_time_seconds < 0:
            raise ValueError("mcts_time_seconds must be non-negative")
        resolved_device = torch.device(device)
        if checkpoint is not None:
            with torch.serialization.safe_globals([WindowsPath]):
                state = torch.load(checkpoint, map_location=resolved_device, weights_only=True)
            config = state.get("config", {}) if isinstance(state, dict) else {}
            channels = int(config.get("channels", 64))
            blocks = int(config.get("blocks", 4))
            value_head = bool(config.get("value_head", False))
            input_channels = int(config.get("input_channels", 15))
            if current_view is None:
                current_view = bool(config.get("current_view", False))
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model = ResNet(
                channels=channels,
                blocks=blocks,
                value_head=value_head,
                input_channels=input_channels,
            )
            model.load_state_dict(state)
        else:
            model = ResNet()
            input_channels = 15
            value_head = False
        if current_view is None:
            current_view = False
        model.to(resolved_device)
        model.eval()
        return cls(
            name=name,
            model=model,
            device=resolved_device,
            sampling_temperature=sampling_temperature,
            sampling_top_k=sampling_top_k,
            current_view=current_view,
            input_channels=input_channels,
            value_head=value_head,
            mcts_time_seconds=mcts_time_seconds if value_head else 0.0,
        )

    def choose_move(self, position: Position, position_counts: dict[Position, int] | None = None) -> Move:
        if self.mcts_time_seconds > 0 and self.value_head:
            return self._choose_mcts(position, position_counts)
        return self._choose_policy_move(position, position_counts)

    def _model_outputs(self, position: Position) -> tuple[torch.Tensor, torch.Tensor, float]:
        board = position_to_tensor(position)
        if self.current_view:
            board = current_view_tensor(position, board)
            if self.input_channels == 14:
                board = board[:14]
        with torch.no_grad():
            outputs = self.model(board.unsqueeze(0).to(self.device))
        value = float(outputs[2].reshape(-1)[0].item()) if self.value_head else 0.0
        return outputs[0][0], outputs[1][0], max(-1.0, min(1.0, value))

    def _policy_priors(self, position: Position, legal: list[Move]) -> tuple[dict[Move, float], float]:
        start_logits, end_logits, value = self._model_outputs(position)
        priors: dict[Move, float] = {}
        scores = []
        for move in legal:
            model_start = current_view_index(move.start) if self.current_view and position.side_to_move == "b" else move.start
            model_end = current_view_index(move.end) if self.current_view and position.side_to_move == "b" else move.end
            scores.append(start_logits[model_start] + end_logits[model_end])
        probabilities = torch.softmax(torch.stack(scores), dim=0).cpu().tolist()
        for move, probability in zip(legal, probabilities):
            priors[move] = float(probability)
        return priors, value

    def _choose_mcts(self, position: Position, position_counts: dict[Position, int] | None) -> Move:
        root_position = position

        def search_legal(node_position: Position) -> list[Move]:
            moves = legal_moves(node_position)
            if node_position != root_position or not position_counts:
                return moves
            fresh_moves = [
                move
                for move in moves
                if position_counts.get(apply_move(node_position, move), 0) == 0
            ]
            return fresh_moves or moves

        def terminal_value(node_position: Position) -> float | None:
            if not king_exists(node_position, "w"):
                return 1.0 if node_position.side_to_move == "b" else -1.0
            if not king_exists(node_position, "b"):
                return 1.0 if node_position.side_to_move == "w" else -1.0
            if not legal_moves(node_position):
                return -1.0
            return None

        def policy_value(node_position: Position, node_legal: list[Move]) -> tuple[dict[Move, float], float]:
            return self._policy_priors(node_position, node_legal)

        search = MCTS(
            legal_moves=search_legal,
            apply_move=apply_move,
            policy_value=policy_value,
            terminal_value=terminal_value,
        )
        opening_ply = max(len(position_counts or {}) - 1, 0)
        temperature = self.opening_temperature if opening_ply < self.opening_plies else 0.0
        return search.search(position, self.mcts_time_seconds, root_temperature=temperature)

    def _choose_policy_move(self, position: Position, position_counts: dict[Position, int] | None = None) -> Move:
        legal = legal_moves(position)
        if not legal:
            raise ValueError("no legal moves available")

        board = position_to_tensor(position)
        model_legal = [(move.start, move.end) for move in legal]
        if self.current_view:
            board = current_view_tensor(position, board)
            if self.input_channels == 14:
                board = board[:14]
            if position.side_to_move == "b":
                model_legal = [
                    (current_view_index(move.start), current_view_index(move.end))
                    for move in legal
                ]
        board = board.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(board)
            start_logits, end_logits = outputs[:2]
            move_logits = joint_move_logits(start_logits, end_logits)
            move_mask = legal_move_mask(
                [model_legal],
                device=move_logits.device,
            )
            masked = apply_legal_move_mask(move_logits, move_mask)
            if position_counts:
                fresh_mask = move_mask.clone()
                for model_start, model_end in model_legal:
                    original_move = Move(
                        current_view_index(model_start) if self.current_view and position.side_to_move == "b" else model_start,
                        current_view_index(model_end) if self.current_view and position.side_to_move == "b" else model_end,
                    )
                    next_position = apply_move(position, original_move)
                    if position_counts.get(next_position, 0) > 0:
                        fresh_mask[0, model_start * 90 + model_end] = False
                if fresh_mask.any():
                    masked = masked.masked_fill(~fresh_mask, torch.finfo(masked.dtype).min)
            opening_ply = max(len(position_counts or {}) - 1, 0)
            temperature = (
                self.opening_temperature
                if opening_ply < self.opening_plies
                else self.sampling_temperature
            )
            if temperature is None:
                selected = int(masked.argmax(dim=1).item())
            else:
                candidate_count = min(self.sampling_top_k, len(legal))
                candidate_logits, candidate_indices = masked.topk(candidate_count, dim=1)
                probabilities = torch.softmax(candidate_logits / temperature, dim=1)
                sampled_rank = int(torch.multinomial(probabilities, 1).item())
                selected = int(candidate_indices[0, sampled_rank].item())
        model_move = Move(selected // 90, selected % 90)
        move = (
            Move(current_view_index(model_move.start), current_view_index(model_move.end))
            if self.current_view and position.side_to_move == "b"
            else model_move
        )
        if (move.start, move.end) not in {(m.start, m.end) for m in legal}:
            raise RuntimeError(f"model selected illegal move: {move_to_iccs(move)}")
        return move


class PikafishPlayer:
    """UCI adapter for Pikafish's built-in NNUE alpha-beta search."""

    def __init__(
        self,
        *,
        name: str,
        command: str,
        move_time_ms: int = 1000,
        threads: int | None = None,
        hash_mb: int | None = None,
    ) -> None:
        if move_time_ms < 1:
            raise ValueError("Pikafish move time must be positive")
        self.name = name
        self.command = command
        self.move_time_ms = move_time_ms
        self.threads = threads
        self.hash_mb = hash_mb
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, *, name: str) -> "PikafishPlayer":
        command = pikafish_command()
        if command is None:
            raise ValueError("Pikafish executable not found; set PIKAFISH_PATH or add pikafish to PATH")
        return cls(
            name=name,
            command=command,
            move_time_ms=int(os.environ.get("PIKAFISH_MOVE_TIME_MS", "1000")),
            threads=_positive_environment_int("PIKAFISH_THREADS"),
            hash_mb=_positive_environment_int("PIKAFISH_HASH_MB"),
        )

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._lines = queue.Queue()
        engine_path = Path(self.command)
        install_dir = engine_path.parent.parent
        working_dir = install_dir if (install_dir / "pikafish.nnue").is_file() else None
        self._process = subprocess.Popen(
            [self.command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self._process.stdout is not None
        threading.Thread(target=self._read_output, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok")
        if self.threads is not None:
            self._send(f"setoption name Threads value {self.threads}")
        if self.hash_mb is not None:
            self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        self._wait_for("readyok")

    def _read_output(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line.strip())
        self._lines.put(None)

    def _send(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Pikafish process is not running")
        self._process.stdin.write(f"{command}\n")
        self._process.stdin.flush()

    def _wait_for(self, expected: str) -> str:
        while True:
            try:
                line = self._lines.get(timeout=10)
            except queue.Empty as error:
                raise RuntimeError(f"Pikafish timed out waiting for {expected}") from error
            if line is None:
                raise RuntimeError("Pikafish exited unexpectedly")
            if line == expected or line.startswith(f"{expected} "):
                return line

    def choose_move(self, position: Position, position_counts: dict[Position, int] | None = None) -> Move:
        legal = legal_moves(position)
        if not legal:
            raise ValueError("no legal moves available")
        with self._lock:
            self._start()
            self._send(f"position fen {position_to_fen(position)}")
            self._send(f"go movetime {self.move_time_ms}")
            response = self._wait_for("bestmove").split()
        if len(response) < 2 or response[1] == "(none)":
            raise RuntimeError("Pikafish returned no move")
        uci_move = response[1]
        if len(uci_move) == 4:
            uci_move = f"{uci_move[:2]}-{uci_move[2:]}"
        try:
            move = iccs_to_move(uci_move)
        except ValueError as error:
            raise RuntimeError(f"invalid Pikafish move: {response[1]}") from error
        if move not in legal:
            raise RuntimeError(f"Pikafish selected illegal move: {move_to_iccs(move)}")
        return move


def _positive_environment_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed
