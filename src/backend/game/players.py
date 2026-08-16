from __future__ import annotations

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
    sampling_temperature: float | None = None
    sampling_top_k: int = 5

    @classmethod
    def from_checkpoint(
        cls,
        *,
        name: str,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        sampling_temperature: float | None = None,
        sampling_top_k: int = 5,
    ) -> "ModelPlayer":
        if sampling_temperature is not None and sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if sampling_top_k < 1:
            raise ValueError("sampling_top_k must be positive")
        resolved_device = torch.device(device)
        if checkpoint is not None:
            with torch.serialization.safe_globals([WindowsPath]):
                state = torch.load(checkpoint, map_location=resolved_device, weights_only=True)
            config = state.get("config", {}) if isinstance(state, dict) else {}
            channels = int(config.get("channels", 64))
            blocks = int(config.get("blocks", 4))
            value_head = bool(config.get("value_head", False))
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model = ResNet(channels=channels, blocks=blocks, value_head=value_head)
            model.load_state_dict(state)
        else:
            model = ResNet()
        model.to(resolved_device)
        model.eval()
        return cls(
            name=name,
            model=model,
            device=resolved_device,
            sampling_temperature=sampling_temperature,
            sampling_top_k=sampling_top_k,
        )

    def choose_move(self, position: Position, position_counts: dict[Position, int] | None = None) -> Move:
        legal = legal_moves(position)
        if not legal:
            raise ValueError("no legal moves available")

        board = position_to_tensor(position).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(board)
            start_logits, end_logits = outputs[:2]
            move_logits = joint_move_logits(start_logits, end_logits)
            move_mask = legal_move_mask(
                [[(move.start, move.end) for move in legal]],
                device=move_logits.device,
            )
            masked = apply_legal_move_mask(move_logits, move_mask)
            if position_counts:
                fresh_mask = move_mask.clone()
                for move in legal:
                    next_position = apply_move(position, move)
                    if position_counts.get(next_position, 0) > 0:
                        fresh_mask[0, move.start * 90 + move.end] = False
                if fresh_mask.any():
                    masked = masked.masked_fill(~fresh_mask, torch.finfo(masked.dtype).min)
            if self.sampling_temperature is None:
                selected = int(masked.argmax(dim=1).item())
            else:
                candidate_count = min(self.sampling_top_k, len(legal))
                candidate_logits, candidate_indices = masked.topk(candidate_count, dim=1)
                probabilities = torch.softmax(candidate_logits / self.sampling_temperature, dim=1)
                sampled_rank = int(torch.multinomial(probabilities, 1).item())
                selected = int(candidate_indices[0, sampled_rank].item())
        move = Move(selected // 90, selected % 90)
        if (move.start, move.end) not in {(m.start, m.end) for m in legal}:
            raise RuntimeError(f"model selected illegal move: {move_to_iccs(move)}")
        return move
