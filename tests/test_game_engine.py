from __future__ import annotations

import pytest

from backend.game.engine import Move, apply_move, is_theoretical_draw, legal_moves, move_to_iccs, parse_fen, resets_natural_limit


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


def test_natural_limit_resets_on_capture_or_pawn_crossing() -> None:
    quiet_position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    assert not resets_natural_limit(quiet_position, Move(13, 22))

    capture_position = parse_fen("4k4/9/9/9/9/9/9/9/4r4/4K4 w - - 0 1")
    assert resets_natural_limit(capture_position, Move(13, 4))

    pawn_position = parse_fen("4k4/9/9/9/9/4P4/9/9/9/4K4 w - - 0 1")
    assert resets_natural_limit(pawn_position, Move(40, 49))


def test_theoretical_draw_detects_only_advisors_and_elephants() -> None:
    position = parse_fen("3k5/9/9/9/9/9/9/9/3A5/4K4 w - - 0 1")
    material_position = parse_fen("3k5/9/9/9/9/9/9/9/3R5/4K4 w - - 0 1")

    assert is_theoretical_draw(position)
    assert not is_theoretical_draw(material_position)


def test_repeated_position_is_drawn_on_third_occurrence() -> None:
    pytest.importorskip("flask")
    from backend.app import _evaluate_result

    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")

    assert _evaluate_result(position, {position: 2}) is None
    assert _evaluate_result(position, {position: 3}) == "draw_repetition"


def test_maximum_ply_limit_is_drawn() -> None:
    pytest.importorskip("flask")
    from backend.app import MAX_PLIES, _evaluate_result

    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")

    assert _evaluate_result(position, quiet_plies=0, total_plies=MAX_PLIES) == "draw_move_limit"


def test_repeated_cycle_assigns_long_check_to_checking_side() -> None:
    pytest.importorskip("flask")
    from backend.app import WebGame, _cycle_violation

    position = parse_fen("3k5/9/9/9/9/9/9/9/3A5/4K4 w - - 0 1")
    other_position = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    game = WebGame(
        game_id="long-check",
        mode="model-model",
        human_side=None,
        initial_position=position,
        position=position,
        players={"w": object(), "b": object()},
        position_counts={position: 3},
        position_history=[position, other_position, position, other_position, position],
        move_history=[Move(0, 1)] * 4,
        mover_sides=["w", "b", "w", "b"],
        checking_sides=["w", None, "w", None],
        chasing_targets=[set()] * 4,
    )

    assert _cycle_violation(game) == "black_win_long_check"


def test_repeated_cycle_assigns_long_chase_to_attacking_side() -> None:
    pytest.importorskip("flask")
    from backend.app import WebGame, _cycle_violation

    position = parse_fen("3k5/9/9/9/9/9/9/9/3A5/4K4 w - - 0 1")
    other_position = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    target = (3, 4, "A")
    game = WebGame(
        game_id="long-chase",
        mode="model-model",
        human_side=None,
        initial_position=position,
        position=position,
        players={"w": object(), "b": object()},
        position_counts={position: 3},
        position_history=[position, other_position, position, other_position, position],
        move_history=[Move(0, 1)] * 4,
        mover_sides=["w", "b", "w", "b"],
        checking_sides=[None] * 4,
        chasing_targets=[{target}, set(), {target}, set()],
    )

    assert _cycle_violation(game) == "black_win_long_chase"


def test_model_match_stops_after_third_repeated_position() -> None:
    pytest.importorskip("flask")
    from backend.app import WebGame, _apply_one_model_move

    position = parse_fen("4k4/8r/9/9/4P4/9/9/9/R8/4K4 w - - 0 1")

    class _CyclePlayer:
        def __init__(self, moves: list[Move]) -> None:
            self.moves = iter(moves)

        def choose_move(self, _position: object, _position_counts: object = None) -> Move:
            return next(self.moves)

    red_moves = [Move(9, 18), Move(18, 9)] * 2
    black_moves = [Move(80, 71), Move(71, 80)] * 2
    game = WebGame(
        game_id="cycle",
        mode="model-model",
        human_side=None,
        initial_position=position,
        position=position,
        players={"w": _CyclePlayer(red_moves), "b": _CyclePlayer(black_moves)},
        position_counts={position: 1},
        position_history=[position],
    )

    for _ in range(7):
        assert _apply_one_model_move(game)
        assert game.result is None

    assert _apply_one_model_move(game)
    assert game.result == "draw_repetition"
    assert len(game.move_history) == 8
    assert len(game.position_history) == 9
    assert len(game.checking_sides) == 8


def test_model_player_avoids_move_to_seen_position() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from backend.game.players import ModelPlayer

    class _DeterministicModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = inputs.shape[0]
            start = torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device)
            end = torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device)
            start[:, 13] = 12.0
            end[:, 22] = 12.0
            return start, end

    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    repeated_move = Move(13, 22)
    repeated_position = apply_move(position, repeated_move)
    player = ModelPlayer(name="TestModel", model=_DeterministicModel(), device=torch.device("cpu"))

    selected = player.choose_move(position, {repeated_position: 1})

    assert selected != repeated_move


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


def test_model_player_samples_only_legal_top_k_moves() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from backend.game.players import ModelPlayer

    class _FlatModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = inputs.shape[0]
            return (
                torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device),
                torch.zeros((batch, 90), dtype=torch.float32, device=inputs.device),
            )

    position = parse_fen("4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    player = ModelPlayer(
        name="Sampled",
        model=_FlatModel(),
        device=torch.device("cpu"),
        sampling_temperature=0.3,
        sampling_top_k=5,
    )
    legal = {(move.start, move.end) for move in legal_moves(position)}

    for _ in range(10):
        move = player.choose_move(position)
        assert (move.start, move.end) in legal
