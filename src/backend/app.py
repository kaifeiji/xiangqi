from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory

from backend.game.engine import (
    START_FEN,
    Position,
    apply_move,
    iccs_to_move,
    is_in_check,
    king_exists,
    legal_moves,
    move_to_iccs,
    parse_fen,
)

MODEL_SUFFIXES = {".ckpt", ".pt", ".pth"}
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
WEB_CLIENT_DIR = PROJECT_DIR
WEB_DIST_DIR = PROJECT_DIR / "dist"


@dataclass
class WebGame:
    game_id: str
    mode: str
    human_side: str | None
    position: Position
    players: dict[str, Any]
    turn: int = 1
    result: str | None = None
    last_error: str | None = None


def _evaluate_result(position: Position) -> str | None:
    if not king_exists(position, "w"):
        return "black_win"
    if not king_exists(position, "b"):
        return "red_win"
    side = position.side_to_move
    available = legal_moves(position)
    if not available:
        if is_in_check(position, side):
            return "black_win" if side == "w" else "red_win"
        return "draw_stalemate"
    return None


def _is_model_turn(game: WebGame) -> bool:
    return game.players[game.position.side_to_move] is not None


def _apply_one_model_move(game: WebGame) -> bool:
    if game.result is not None or not _is_model_turn(game):
        return False
    player = game.players[game.position.side_to_move]
    if player is None:
        return False
    move = player.choose_move(game.position)
    game.position = apply_move(game.position, move)
    game.turn += 1
    game.result = _evaluate_result(game.position)
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
        "result": game.result,
        "in_check": game.result is None and is_in_check(game.position, side),
        "legal_moves": sorted(legal_set),
        "is_human_turn": game.players[side] is None and game.result is None,
        "board": board,
        "last_error": game.last_error,
    }


def _available_models() -> list[dict[str, str]]:
    if not MODELS_DIR.is_dir():
        return []
    return [
        {"id": path.relative_to(MODELS_DIR).as_posix(), "name": path.name}
        for path in sorted(MODELS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]


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


def create_app(dev_web_url: str | None = None) -> Flask:
    app = Flask(__name__)
    games: dict[str, WebGame] = {}
    lock = threading.Lock()

    def create_model(name: str, model_id: Any) -> Any:
        from backend.game.players import ModelPlayer

        return ModelPlayer.from_checkpoint(name=name, checkpoint=_resolve_model(model_id), device="cpu")

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
        fen = payload.get("fen", START_FEN)
        model = payload.get("model")
        red_model = payload.get("red_model")
        black_model = payload.get("black_model")

        try:
            position = parse_fen(fen)
            players: dict[str, Any]
            if mode == "human-model":
                if human_side not in {"w", "b"}:
                    return jsonify({"error": "invalid human_side"}), 400
                model_side = "b" if human_side == "w" else "w"
                players = {
                    human_side: None,
                    model_side: create_model("Model", model),
                }
            else:
                players = {
                    "w": create_model("RedModel", red_model),
                    "b": create_model("BlackModel", black_model),
                }
                human_side = None
        except Exception as error:
            return jsonify({"error": str(error)}), 400

        game = WebGame(
            game_id=str(uuid.uuid4()),
            mode=mode,
            human_side=human_side,
            position=position,
            players=players,
        )
        game.result = _evaluate_result(game.position)
        with lock:
            games[game.game_id] = game
        return jsonify(_serialize_game(game))

    @app.get("/api/games/<game_id>")
    def get_game(game_id: str) -> Any:
        with lock:
            game = games.get(game_id)
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
                game.position = apply_move(game.position, move)
                game.turn += 1
                game.result = _evaluate_result(game.position)
                game.last_error = None
            except Exception as error:
                game.last_error = str(error)
                return jsonify({"error": str(error)}), 400
            return jsonify(_serialize_game(game))

    @app.post("/api/games/<game_id>/step")
    def step_model(game_id: str) -> Any:
        with lock:
            game = games.get(game_id)
            if game is None:
                return jsonify({"error": "game not found"}), 404
            if game.result is not None:
                return jsonify({"error": "game already finished"}), 400
            if not _is_model_turn(game):
                return jsonify({"error": "current side is controlled by human"}), 400
            try:
                moved = _apply_one_model_move(game)
                if not moved:
                    return jsonify({"error": "no model move applied"}), 400
                game.last_error = None
            except Exception as error:
                game.last_error = str(error)
                return jsonify({"error": str(error)}), 400
            return jsonify(_serialize_game(game))

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
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=not args.dev)
    finally:
        if dev_server is not None:
            dev_server.terminate()
            dev_server.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
