from __future__ import annotations
from pathlib import Path, WindowsPath

import pytest
import torch

from backend.game.players import ModelPlayer
from backend.models import PikafishResNet, ResNet


def test_web_lists_models_from_models_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("flask")
    import backend.app as app_module
    import backend.game.players as players_module

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "first.pt").touch()
    (models_dir / "nested").mkdir()
    (models_dir / "nested" / "second.ckpt").touch()
    (models_dir / "ignore.txt").touch()
    monkeypatch.setattr(app_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(players_module, "pikafish_command", lambda: None)

    app = app_module.create_app()
    client = app.test_client()

    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.get_json()["models"] == [
        {"id": "first.pt", "name": "first.pt"},
        {"id": "nested/second.ckpt", "name": "second.ckpt"},
    ]


def test_web_lists_pikafish_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("flask")
    import backend.app as app_module
    import backend.game.players as players_module

    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(players_module, "pikafish_command", lambda: "C:/tools/pikafish.exe")

    app = app_module.create_app()
    response = app.test_client().get("/api/models")

    assert response.status_code == 200
    assert response.get_json()["models"] == [
        {"id": "pikafish", "name": "Pikafish (NNUE + alpha-beta)"}
    ]


def test_web_rejects_missing_selected_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("flask")
    import backend.app as app_module

    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path / "models")
    app = app_module.create_app()
    client = app.test_client()

    created = client.post("/api/games", json={"mode": "human-model", "model": "missing.pt"})
    assert created.status_code == 400
    assert created.get_json()["error"] == "model not found"


def test_model_player_loads_training_checkpoint_with_windows_path_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model": ResNet(channels=96, blocks=6).state_dict(),
            "config": {"checkpoint_dir": WindowsPath("checkpoints/run-1"), "channels": 96, "blocks": 6},
        },
        checkpoint_path,
    )

    player = ModelPlayer.from_checkpoint(name="Model", checkpoint=checkpoint_path)

    assert isinstance(player.model, ResNet)


def test_model_player_loads_pikafish_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "pikafish-model.pt"
    torch.save(
        {
            "model": PikafishResNet(channels=32, blocks=2).state_dict(),
            "config": {"checkpoint_dir": WindowsPath("checkpoints/pikafish-run")},
        },
        checkpoint_path,
    )

    player = ModelPlayer.from_checkpoint(name="PikafishModel", checkpoint=checkpoint_path)

    assert isinstance(player.model, PikafishResNet)
    assert player.current_view is True


def test_select_initial_fen_uses_opening_pool_by_default() -> None:
    pytest.importorskip("flask")
    import backend.app as app_module

    openings = (
        "1Cbakabnr/9/rc5c1/p1p1p1p1p/9/2P6/P3P1P1P/7C1/9/RNBAKABNR b - - 2 2",
    )
    fen = app_module._select_initial_fen({}, openings)

    assert fen == openings[0]


def test_select_initial_fen_falls_back_to_start_fen_without_opening_pool() -> None:
    pytest.importorskip("flask")
    import backend.app as app_module

    assert app_module._select_initial_fen({}, ()) == app_module.START_FEN


def test_curated_opening_positions_are_non_empty_and_black_to_move() -> None:
    pytest.importorskip("flask")
    import backend.opening_book as opening_book_module

    positions = opening_book_module.curated_opening_positions()

    assert positions
    assert len(positions) < 44
    assert all(position.split()[1] == "b" for position in positions)
