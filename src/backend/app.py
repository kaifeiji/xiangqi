from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory
from backend.opening_book import curated_opening_positions

from backend.game.engine import (
    START_FEN,
    Move,
    Position,
    apply_move,
    attacking_targets,
    iccs_to_move,
    is_in_check,
    is_theoretical_draw,
    king_exists,
    legal_moves,
    move_to_iccs,
    parse_fen,
    resets_natural_limit,
)

MODEL_SUFFIXES = {".ckpt", ".pt", ".pth"}
PIKAFISH_MODEL_ID = "pikafish"
NATURAL_LIMIT_PLIES = 120
MAX_PLIES = 600
GAME_IDLE_TIMEOUT_SECONDS = 600.0
GAME_CLEANUP_INTERVAL_SECONDS = 60.0
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
WEB_CLIENT_DIR = PROJECT_DIR
WEB_DIST_DIR = PROJECT_DIR / "dist"


@dataclass
class WebGame:
    game_id: str
    mode: str
    human_side: str | None
    initial_position: Position
    position: Position
    players: dict[str, Any]
    turn: int = 1
    result: str | None = None
    last_error: str | None = None
    position_counts: dict[Position, int] = field(default_factory=dict)
    quiet_plies: int = 0
    position_history: list[Position] = field(default_factory=list)
    move_history: list[Move] = field(default_factory=list)
    checking_sides: list[str | None] = field(default_factory=list)
    mover_sides: list[str] = field(default_factory=list)
    chasing_targets: list[set[tuple[int, int, str]]] = field(default_factory=list)
    last_accessed_at: float = field(default_factory=time.monotonic)


def _evaluate_result(
    position: Position,
    position_counts: dict[Position, int] | None = None,
    quiet_plies: int = 0,
    total_plies: int = 0,
) -> str | None:
    if not king_exists(position, "w"):
        return "black_win"
    if not king_exists(position, "b"):
        return "red_win"
    side = position.side_to_move
    available = legal_moves(position)
    if not available:
        # Xiangqi rules: no legal move (including stalemate) is a loss for side to move.
        return "black_win" if side == "w" else "red_win"
    if is_theoretical_draw(position):
        return "draw_insufficient_material"
    if position_counts is not None and position_counts.get(position, 0) >= 3:
        return "draw_repetition"
    if quiet_plies >= NATURAL_LIMIT_PLIES:
        return "draw_natural_limit"
    if total_plies >= MAX_PLIES:
        return "draw_move_limit"
    return None


def _record_position(game: WebGame) -> None:
    game.position_counts[game.position] = game.position_counts.get(game.position, 0) + 1


def _cycle_violation(game: WebGame) -> str | None:
    if game.position_counts.get(game.position, 0) < 3:
        return None
    occurrences = [index for index, position in enumerate(game.position_history) if position == game.position]
    if len(occurrences) < 3:
        return None
    cycle_start = occurrences[-3]
    cycle_end = len(game.move_history)
    cycle_moves = range(cycle_start, cycle_end)
    for offender in ("w", "b"):
        offender_moves = [index for index in cycle_moves if game.mover_sides[index] == offender]
        if len(offender_moves) >= 2 and all(game.checking_sides[index] == offender for index in offender_moves):
            return "black_win_long_check" if offender == "w" else "red_win_long_check"
        target_sets = [game.chasing_targets[index] for index in offender_moves]
        if len(target_sets) >= 2:
            common_targets = set.intersection(*target_sets)
            if common_targets:
                return "black_win_long_chase" if offender == "w" else "red_win_long_chase"
    return None


def _is_model_turn(game: WebGame) -> bool:
    return game.players[game.position.side_to_move] is not None


def _apply_move_to_game(game: WebGame, move: Any) -> None:
    moving_side = game.position.side_to_move
    resets_limit = resets_natural_limit(game.position, move)
    game.position = apply_move(game.position, move)
    game.turn += 1
    game.quiet_plies = 0 if resets_limit else game.quiet_plies + 1
    game.move_history.append(move)
    game.position_history.append(game.position)
    game.mover_sides.append(moving_side)
    if king_exists(game.position, "w") and king_exists(game.position, "b"):
        checked_side = game.position.side_to_move
        game.checking_sides.append(moving_side if is_in_check(game.position, checked_side) else None)
        game.chasing_targets.append(attacking_targets(game.position, moving_side))
    else:
        game.checking_sides.append(None)
        game.chasing_targets.append(set())
    _record_position(game)
    game.result = _cycle_violation(game)
    if game.result is None:
        game.result = _evaluate_result(game.position, game.position_counts, game.quiet_plies, game.turn)


def _apply_one_model_move(game: WebGame) -> bool:
    if game.result is not None or not _is_model_turn(game):
        return False
    player = game.players[game.position.side_to_move]
    if player is None:
        return False
    move = player.choose_move(game.position, game.position_counts)
    _apply_move_to_game(game, move)
    return True


def _rebuild_game_state(game: WebGame) -> None:
    replay_moves = list(game.move_history)
    game.position = game.initial_position
    game.turn = 1
    game.result = None
    game.last_error = None
    game.quiet_plies = 0
    game.position_counts = {game.position: 1}
    game.position_history = [game.position]
    game.move_history = []
    game.checking_sides = []
    game.mover_sides = []
    game.chasing_targets = []

    for move in replay_moves:
        _apply_move_to_game(game, move)


def _undo_last_plies(game: WebGame, plies: int) -> bool:
    if game.result is not None or plies < 1 or not game.move_history:
        return False
    keep_moves = max(len(game.move_history) - plies, 0)
    game.move_history = game.move_history[:keep_moves]
    _rebuild_game_state(game)
    return True


def _piece_name(piece: str | None) -> str:
    if piece is None:
        return ""
    names = {
        "K": "帅",
        "A": "仕",
        "B": "相",
        "N": "马",
        "R": "车",
        "C": "炮",
        "P": "兵",
        "k": "将",
        "a": "士",
        "b": "象",
        "n": "马",
        "r": "车",
        "c": "炮",
        "p": "卒",
    }
    return names[piece]


def _serialize_game(game: WebGame) -> dict[str, Any]:
    legal = legal_moves(game.position) if game.result is None else []
    legal_set = {move_to_iccs(move) for move in legal}
    board = []
    for row in range(9, -1, -1):
        row_cells = []
        for col in range(9):
            piece = game.position.board[row][col]
            row_cells.append(
                {
                    "piece": piece,
                    "label": _piece_name(piece),
                    "square": f"{chr(ord('A') + col)}{row}",
                }
            )
        board.append(row_cells)

    side = game.position.side_to_move
    return {
        "game_id": game.game_id,
        "mode": game.mode,
        "human_side": game.human_side,
        "side_to_move": side,
        "turn": game.turn,
        "quiet_plies": game.quiet_plies,
        "result": game.result,
        "in_check": game.result is None and is_in_check(game.position, side),
        "legal_moves": sorted(legal_set),
        "is_human_turn": game.players[side] is None and game.result is None,
        "board": board,
        "last_error": game.last_error,
    }


def _available_models() -> list[dict[str, str]]:
    models = []
    if MODELS_DIR.is_dir():
        models = [
        {"id": path.relative_to(MODELS_DIR).as_posix(), "name": path.name}
        for path in sorted(MODELS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
        ]
    from backend.game.players import pikafish_command

    if pikafish_command() is not None:
        models.append({"id": PIKAFISH_MODEL_ID, "name": "Pikafish (NNUE + alpha-beta)"})
    return models


def _resolve_model(model_id: Any) -> Path:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model is required")
    candidate = (MODELS_DIR / model_id).resolve()
    try:
        candidate.relative_to(MODELS_DIR.resolve())
    except ValueError as error:
        raise ValueError("invalid model") from error
    if not candidate.is_file() or candidate.suffix.lower() not in MODEL_SUFFIXES:
        raise ValueError("model not found")
    return candidate


def _select_initial_fen(payload: dict[str, Any], opening_positions: tuple[str, ...]) -> str:
    requested_fen = payload.get("fen")
    if isinstance(requested_fen, str) and requested_fen.strip():
        return requested_fen
    if not opening_positions:
        return START_FEN
    return random.choice(opening_positions)


def create_app(dev_web_url: str | None = None) -> Flask:
    app = Flask(__name__)
    games: dict[str, WebGame] = {}
    lock = threading.Lock()
    opening_positions = curated_opening_positions()

    def cleanup_games() -> None:
        while True:
            time.sleep(GAME_CLEANUP_INTERVAL_SECONDS)
            cutoff = time.monotonic() - GAME_IDLE_TIMEOUT_SECONDS
            with lock:
                stale_ids = [
                    game_id
                    for game_id, game in games.items()
                    if game.last_accessed_at < cutoff
                ]
                for game_id in stale_ids:
                    games.pop(game_id, None)
            if stale_ids:
                app.logger.info("cleaned up %d idle games", len(stale_ids))

    threading.Thread(target=cleanup_games, name="game-cleanup", daemon=True).start()
    preferred_device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            preferred_device = "cuda"
    except Exception:
        preferred_device = "cpu"
    app.logger.info("model inference device: %s", preferred_device)

    def create_model(
        name: str,
        model_id: Any,
        *,
        mcts_time_seconds: float = 0.0,
    ) -> Any:
        from backend.game.players import ModelPlayer, PikafishPlayer

        if model_id == PIKAFISH_MODEL_ID:
            return PikafishPlayer.from_environment(name=name)

        return ModelPlayer.from_checkpoint(
            name=name,
            checkpoint=_resolve_model(model_id),
            device=preferred_device,
            mcts_time_seconds=mcts_time_seconds,
        )

    @app.get("/api/models")
    def list_models() -> Any:
        return jsonify({"models": _available_models()})

    @app.post("/api/games")
    def create_game() -> Any:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", "human-model")
        if mode not in {"human-model", "model-model"}:
            return jsonify({"error": "invalid mode"}), 400
        human_side = payload.get("human_side", "w")
        model = payload.get("model")
        red_model = payload.get("red_model")
        black_model = payload.get("black_model")
        try:
            mcts_time_seconds = float(payload.get("mcts_time_seconds", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "mcts_time_seconds must be a number"}), 400
        if mcts_time_seconds not in {0.0, 1.0, 3.0, 5.0, 10.0}:
            return jsonify({"error": "mcts_time_seconds must be one of 0, 1, 3, 5, 10"}), 400

        try:
            fen = _select_initial_fen(payload, opening_positions)
            position = parse_fen(fen)
            players: dict[str, Any]
            if mode == "human-model":
                if human_side not in {"w", "b"}:
                    return jsonify({"error": "invalid human_side"}), 400
                model_side = "b" if human_side == "w" else "w"
                players = {
                    human_side: None,
                    model_side: create_model("Model", model, mcts_time_seconds=mcts_time_seconds),
                }
            else:
                players = {
                    "w": create_model("RedModel", red_model, mcts_time_seconds=mcts_time_seconds),
                    "b": create_model("BlackModel", black_model, mcts_time_seconds=mcts_time_seconds),
                }
                human_side = None
        except Exception as error:
            return jsonify({"error": str(error)}), 400

        game = WebGame(
            game_id=str(uuid.uuid4()),
            mode=mode,
            human_side=human_side,
            initial_position=position,
            position=position,
            players=players,
            position_counts={position: 1},
            position_history=[position],
        )
        game.result = _evaluate_result(game.position, game.position_counts, game.quiet_plies, game.turn)
        with lock:
            games[game.game_id] = game
        return jsonify(_serialize_game(game))

    @app.get("/api/games/<game_id>")
    def get_game(game_id: str) -> Any:
        with lock:
            game = games.get(game_id)
            if game is not None:
                game.last_accessed_at = time.monotonic()
        if game is None:
            return jsonify({"error": "game not found"}), 404
        return jsonify(_serialize_game(game))

    @app.post("/api/games/<game_id>/move")
    def human_move(game_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        move_text = payload.get("move")
        if not isinstance(move_text, str):
            return jsonify({"error": "move is required"}), 400

        with lock:
            game = games.get(game_id)
            if game is None:
                return jsonify({"error": "game not found"}), 404
            game.last_accessed_at = time.monotonic()
            if game.result is not None:
                return jsonify({"error": "game already finished"}), 400
            side = game.position.side_to_move
            if game.players[side] is not None:
                return jsonify({"error": "current side is controlled by model"}), 400
            try:
                move = iccs_to_move(move_text)
                legal_set = {(m.start, m.end) for m in legal_moves(game.position)}
                if (move.start, move.end) not in legal_set:
                    return jsonify({"error": "illegal move"}), 400
                _apply_move_to_game(game, move)
                game.last_error = None
            except Exception as error:
                game.last_error = str(error)
                return jsonify({"error": str(error)}), 400
            return jsonify(_serialize_game(game))

    @app.post("/api/games/<game_id>/step")
    def step_model(game_id: str) -> Any:
        request_started = time.perf_counter()
        with lock:
            game = games.get(game_id)
            if game is None:
                return jsonify({"error": "game not found"}), 404
            game.last_accessed_at = time.monotonic()
            if game.result is not None:
                return jsonify({"error": "game already finished"}), 400
            if not _is_model_turn(game):
                return jsonify({"error": "current side is controlled by human"}), 400
            try:
                apply_started = time.perf_counter()
                moved = _apply_one_model_move(game)
                apply_elapsed_ms = (time.perf_counter() - apply_started) * 1000.0
                if not moved:
                    return jsonify({"error": "no model move applied"}), 400
                serialize_started = time.perf_counter()
                payload = _serialize_game(game)
                serialize_elapsed_ms = (time.perf_counter() - serialize_started) * 1000.0
                total_elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                app.logger.info(
                    "step game=%s turn=%s apply_ms=%.1f serialize_ms=%.1f total_ms=%.1f",
                    game.game_id,
                    game.turn,
                    apply_elapsed_ms,
                    serialize_elapsed_ms,
                    total_elapsed_ms,
                )
                game.last_error = None
            except Exception as error:
                game.last_error = str(error)
                return jsonify({"error": str(error)}), 400
            return jsonify(payload)

    @app.post("/api/games/<game_id>/undo")
    def undo_move(game_id: str) -> Any:
        payload = request.get_json(silent=True) or {}
        requested_plies = payload.get("plies", 1)
        try:
            plies = int(requested_plies)
        except (TypeError, ValueError):
            return jsonify({"error": "plies must be an integer"}), 400
        if plies < 1:
            return jsonify({"error": "plies must be >= 1"}), 400

        with lock:
            game = games.get(game_id)
            if game is None:
                return jsonify({"error": "game not found"}), 404
            game.last_accessed_at = time.monotonic()
            if game.mode != "human-model":
                return jsonify({"error": "undo is only supported in human-model mode"}), 400
            if game.result is not None:
                return jsonify({"error": "game is already finished; undo is not allowed"}), 400
            if not _undo_last_plies(game, plies):
                return jsonify({"error": "no moves to undo"}), 400
            return jsonify(_serialize_game(game))

    @app.post("/api/games/<game_id>/close")
    def close_game(game_id: str) -> Any:
        with lock:
            game = games.pop(game_id, None)
        if game is None:
            return jsonify({"error": "game not found"}), 404
        return jsonify({"closed": True, "game_id": game_id})

    @app.get("/")
    @app.get("/<path:asset_path>")
    def client(asset_path: str = "index.html") -> Any:
        if dev_web_url is not None:
            return redirect(f"{dev_web_url}/{asset_path}" if asset_path else dev_web_url)
        if not WEB_DIST_DIR.is_dir():
            return jsonify({"error": "web client is not built; run xiangqi-play"}), 503
        requested_file = WEB_DIST_DIR / asset_path
        if asset_path and requested_file.is_file():
            return send_from_directory(WEB_DIST_DIR, asset_path)
        return send_from_directory(WEB_DIST_DIR, "index.html")

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and serve the Xiangqi web client and API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dev", action="store_true", help="run the Vite development server with frontend hot updates")
    parser.add_argument("--web-port", type=int, default=5173, help="Vite development server port")
    parser.add_argument("--debug", action="store_true")
    return parser


def _build_web_client() -> None:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to build the web client")
    if not (WEB_CLIENT_DIR / "package.json").is_file():
        raise FileNotFoundError(f"web client package not found: {WEB_CLIENT_DIR}")
    subprocess.run([npm, "ci"], cwd=WEB_CLIENT_DIR, check=True)
    subprocess.run([npm, "run", "build"], cwd=WEB_CLIENT_DIR, check=True)


def _start_web_dev_server(host: str, api_port: int, web_port: int) -> subprocess.Popen[bytes]:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to run the web development server")
    environment = os.environ | {"VITE_API_TARGET": f"http://{host}:{api_port}"}
    return subprocess.Popen(
        [npm, "run", "dev", "--", "--host", host, "--port", str(web_port)],
        cwd=WEB_CLIENT_DIR,
        env=environment,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    dev_server: subprocess.Popen[bytes] | None = None
    if args.dev:
        dev_server = _start_web_dev_server(args.host, args.port, args.web_port)
        print(f"Vite development server: http://{args.host}:{args.web_port}")
    else:
        _build_web_client()
    app = create_app(f"http://{args.host}:{args.web_port}" if args.dev else None)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)
    finally:
        if dev_server is not None:
            dev_server.terminate()
            dev_server.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
