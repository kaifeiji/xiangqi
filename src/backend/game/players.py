from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, WindowsPath
from typing import Protocol

import numpy as np
import torch
from torch import nn

from backend.inference.move_scoring import apply_legal_move_mask, joint_move_logits, legal_move_mask
from backend.models.tiny_resnet import TinyResNet

from .engine import (
    BOARD_COLS,
    BOARD_ROWS,
    Move,
    Position,
    iccs_to_move,
    legal_moves,
    move_to_iccs,
)

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

    @classmethod
    def from_checkpoint(
        cls,
        *,
        name: str,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> "ModelPlayer":
        resolved_device = torch.device(device)
        model = TinyResNet()
        if checkpoint is not None:
            with torch.serialization.safe_globals([WindowsPath]):
                state = torch.load(checkpoint, map_location=resolved_device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
        model.to(resolved_device)
        model.eval()
        return cls(name=name, model=model, device=resolved_device)

    def choose_move(self, position: Position) -> Move:
        legal = legal_moves(position)
        if not legal:
            raise ValueError("no legal moves available")

        board = position_to_tensor(position).unsqueeze(0).to(self.device)
        with torch.no_grad():
            start_logits, end_logits = self.model(board)
            move_logits = joint_move_logits(start_logits, end_logits)
            move_mask = legal_move_mask(
                [[(move.start, move.end) for move in legal]],
                device=move_logits.device,
            )
            masked = apply_legal_move_mask(move_logits, move_mask)
            best = int(masked.argmax(dim=1).item())
        move = Move(best // 90, best % 90)
        if (move.start, move.end) not in {(m.start, m.end) for m in legal}:
            raise RuntimeError(f"model selected illegal move: {move_to_iccs(move)}")
        return move
