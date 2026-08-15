from __future__ import annotations

import pytest

from backend.game.engine import Move, legal_moves, move_to_iccs, parse_fen


def test_start_position_has_expected_legal_moves() -> None:
    position = parse_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
    moves = {move_to_iccs(move) for move in legal_moves(position)}
    assert "A3-A4" in moves
    assert "B0-A2" in moves


def test_legal_moves_filter_out_facing_kings_exposure() -> None:
    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    moves = {move_to_iccs(move) for move in legal_moves(position)}
    assert "E1-D1" not in moves
    assert "E1-E2" in moves


def test_model_player_respects_legal_move_mask() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from backend.game.players import ModelPlayer

    class _DeterministicModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = inputs.shape[0]
            start = torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device)
            end = torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device)
            start[:, 0] = 12.0
            end[:, 1] = 12.0
            start[:, 13] = 6.0
            end[:, 22] = 6.0
            return start, end

    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    player = ModelPlayer(name="TestModel", model=_DeterministicModel(), device=torch.device("cpu"))
    move = player.choose_move(position)
    assert move == Move(13, 22)
