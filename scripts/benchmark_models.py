from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from backend.game.engine import (  # noqa: E402
    START_FEN,
    Move,
    Position,
    apply_move,
    attacking_targets,
    is_in_check,
    is_theoretical_draw,
    king_exists,
    legal_moves,
    parse_fen,
    resets_natural_limit,
)
from backend.game.players import ModelPlayer  # noqa: E402
from backend.opening_book import curated_opening_positions  # noqa: E402


NATURAL_LIMIT_PLIES = 120
MAX_PLIES = 600


@dataclass
class Match:
    position: Position
    position_counts: dict[Position, int] = field(default_factory=dict)
    position_history: list[Position] = field(default_factory=list)
    move_history: list[Move] = field(default_factory=list)
    mover_sides: list[str] = field(default_factory=list)
    checking_sides: list[str | None] = field(default_factory=list)
    chasing_targets: list[set[tuple[int, int, str]]] = field(default_factory=list)
    quiet_plies: int = 0
    result: str | None = None

    def __post_init__(self) -> None:
        self.position_counts[self.position] = 1
        self.position_history.append(self.position)

    def evaluate_result(self) -> str | None:
        if not king_exists(self.position, "w"):
            return "black_win"
        if not king_exists(self.position, "b"):
            return "red_win"
        if not legal_moves(self.position):
            return "black_win" if self.position.side_to_move == "w" else "red_win"
        if is_theoretical_draw(self.position):
            return "draw_insufficient_material"
        if self.position_counts.get(self.position, 0) >= 3:
            return "draw_repetition"
        if self.quiet_plies >= NATURAL_LIMIT_PLIES:
            return "draw_natural_limit"
        if len(self.move_history) >= MAX_PLIES:
            return "draw_move_limit"
        return None

    def cycle_violation(self) -> str | None:
        if self.position_counts.get(self.position, 0) < 3:
            return None
        occurrences = [
            index for index, position in enumerate(self.position_history) if position == self.position
        ]
        if len(occurrences) < 3:
            return None
        cycle_start = occurrences[-3]
        cycle_moves = range(cycle_start, len(self.move_history))
        for offender in ("w", "b"):
            offender_moves = [index for index in cycle_moves if self.mover_sides[index] == offender]
            if len(offender_moves) >= 2 and all(
                self.checking_sides[index] == offender for index in offender_moves
            ):
                return "black_win_long_check" if offender == "w" else "red_win_long_check"
            target_sets = [self.chasing_targets[index] for index in offender_moves]
            if len(target_sets) >= 2:
                common_targets = set.intersection(*target_sets)
                if common_targets:
                    return "black_win_long_chase" if offender == "w" else "red_win_long_chase"
        return None

    def play_move(self, move: Move) -> None:
        moving_side = self.position.side_to_move
        resets_limit = resets_natural_limit(self.position, move)
        self.position = apply_move(self.position, move)
        self.quiet_plies = 0 if resets_limit else self.quiet_plies + 1
        self.move_history.append(move)
        self.position_history.append(self.position)
        self.mover_sides.append(moving_side)
        if king_exists(self.position, "w") and king_exists(self.position, "b"):
            checked_side = self.position.side_to_move
            self.checking_sides.append(
                moving_side if is_in_check(self.position, checked_side) else None
            )
            self.chasing_targets.append(attacking_targets(self.position, moving_side))
        else:
            self.checking_sides.append(None)
            self.chasing_targets.append(set())
        self.position_counts[self.position] = self.position_counts.get(self.position, 0) + 1
        self.result = self.cycle_violation() or self.evaluate_result()


def play_game(red: ModelPlayer, black: ModelPlayer, fen: str) -> str:
    match = Match(parse_fen(fen))
    match.result = match.evaluate_result()
    players = {"w": red, "b": black}
    while match.result is None:
        player = players[match.position.side_to_move]
        move = player.choose_move(match.position, match.position_counts)
        match.play_move(move)
    return match.result


def build_player(name: str, checkpoint: Path, args: argparse.Namespace) -> ModelPlayer:
    return ModelPlayer.from_checkpoint(
        name=name,
        checkpoint=checkpoint,
        device=args.device,
        mcts_time_seconds=args.mcts_time,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="让两个模型进行中国象棋对弈 benchmark")
    parser.add_argument("--model-a", type=Path, required=True, help="模型 A checkpoint 路径")
    parser.add_argument("--model-b", type=Path, required=True, help="模型 B checkpoint 路径")
    parser.add_argument("--games", type=int, default=100, help="对弈盘数，默认 100")
    parser.add_argument("--device", default="cpu", help="PyTorch device，默认 cpu")
    parser.add_argument("--mcts-time", type=float, default=0.0, help="每步 MCTS 秒数，默认关闭")
    parser.add_argument("--fen", help="初始局面 FEN；不提供时默认随机主流开局")
    parser.add_argument("--same-colors", action="store_true", help="不交换双方颜色")
    parser.add_argument("--verbose", action="store_true", help="输出每盘结果")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/model_benchmark.json"),
        help="JSON 结果文件，默认 benchmark/model_benchmark.json",
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if args.mcts_time < 0:
        parser.error("--mcts-time must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.model_a.is_file():
        raise SystemExit(f"model A not found: {args.model_a}")
    if not args.model_b.is_file():
        raise SystemExit(f"model B not found: {args.model_b}")

    model_a = build_player("Model A", args.model_a, args)
    model_b = build_player("Model B", args.model_b, args)
    openings = curated_opening_positions()
    if not openings:
        openings = (START_FEN,)
    scores = {"model_a": 0, "model_b": 0, "draw": 0}
    draw_reasons: dict[str, int] = {}
    game_results: list[dict[str, object]] = []

    for game_number in range(1, args.games + 1):
        a_is_red = args.same_colors or game_number % 2 == 1
        red, black = (model_a, model_b) if a_is_red else (model_b, model_a)
        game_fen = args.fen if args.fen else random.choice(openings)
        result = play_game(red, black, game_fen)
        if result.startswith("draw_"):
            winner = "draw"
            draw_reasons[result] = draw_reasons.get(result, 0) + 1
        elif result in {"red_win", "red_win_long_check", "red_win_long_chase"}:
            winner = "model_a" if a_is_red else "model_b"
        else:
            winner = "model_b" if a_is_red else "model_a"
        scores[winner] += 1
        game_results.append(
            {
                "game": game_number,
                "red": red.name,
                "black": black.name,
                "fen": game_fen,
                "result": result,
                "winner": winner,
            }
        )
        if args.verbose:
            print(f"game={game_number} red={red.name} black={black.name} result={result}", flush=True)

    output = {
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "games": args.games,
        "fen": args.fen or "random_curated_opening",
        "same_colors": args.same_colors,
        "mcts_time_seconds": args.mcts_time,
        "summary": {
            "model_a_wins": scores["model_a"],
            "model_b_wins": scores["model_b"],
            "draws": scores["draw"],
            "draw_reasons": draw_reasons,
        },
        "games_detail": game_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"result_json={args.output}")
    print(f"games={args.games}")
    print(f"model_a_wins={scores['model_a']}")
    print(f"model_b_wins={scores['model_b']}")
    print(f"draws={scores['draw']}")
    if not args.same_colors:
        print(f"model_a_red_games={(args.games + 1) // 2}")
        print(f"model_a_black_games={args.games // 2}")
    if draw_reasons:
        print("draw_reasons=" + ",".join(f"{key}:{value}" for key, value in sorted(draw_reasons.items())))


if __name__ == "__main__":
    main()