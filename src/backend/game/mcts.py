from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar


PositionT = TypeVar("PositionT")
MoveT = TypeVar("MoveT", bound=object)


@dataclass
class MCTSNode(Generic[PositionT, MoveT]):
    position: PositionT
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[MoveT, "MCTSNode[PositionT, MoveT]"] = field(default_factory=dict)
    expanded: bool = False


class MCTS(Generic[PositionT, MoveT]):
    def __init__(
        self,
        *,
        legal_moves: Callable[[PositionT], list[MoveT]],
        apply_move: Callable[[PositionT, MoveT], PositionT],
        policy_value: Callable[[PositionT, list[MoveT]], tuple[dict[MoveT, float], float]],
        terminal_value: Callable[[PositionT], float | None],
        exploration: float = 1.25,
    ) -> None:
        self.legal_moves = legal_moves
        self.apply_move = apply_move
        self.policy_value = policy_value
        self.terminal_value = terminal_value
        self.exploration = exploration

    def search(self, position: PositionT, time_seconds: float, root_temperature: float = 0.0) -> MoveT:
        if root_temperature < 0:
            raise ValueError("root_temperature must be non-negative")
        legal = self.legal_moves(position)
        if not legal:
            raise ValueError("no legal moves available")
        if len(legal) == 1:
            return legal[0]

        root = MCTSNode[PositionT, MoveT](position)
        self._expand(root)
        deadline = time.perf_counter() + time_seconds
        while time.perf_counter() < deadline or root.visits < 1:
            self._simulate(root, {root.position})
        if root_temperature == 0.0:
            return max(root.children.items(), key=lambda item: item[1].visits)[0]
        weighted = [
            (move, max(child.visits, 1) ** (1.0 / root_temperature))
            for move, child in root.children.items()
        ]
        moves, weights = zip(*weighted)
        return random.choices(moves, weights=weights, k=1)[0]

    def _expand(self, node: MCTSNode[PositionT, MoveT]) -> float:
        terminal = self.terminal_value(node.position)
        if terminal is not None:
            node.expanded = True
            return terminal
        legal = self.legal_moves(node.position)
        priors, value = self.policy_value(node.position, legal)
        for move in legal:
            node.children[move] = MCTSNode(
                self.apply_move(node.position, move),
                priors[move],
            )
        node.expanded = True
        return value

    def _simulate(self, node: MCTSNode[PositionT, MoveT], path_positions: set[PositionT]) -> float:
        terminal = self.terminal_value(node.position)
        if terminal is not None:
            value = terminal
        elif not node.expanded:
            value = self._expand(node)
        else:
            parent_visits = max(node.visits, 1)
            _, child = max(
                node.children.items(),
                key=lambda item: (
                    -(item[1].value_sum / item[1].visits) if item[1].visits else 0.0
                )
                + self.exploration
                * item[1].prior
                * math.sqrt(parent_visits)
                / (1 + item[1].visits),
            )
            value = (
                0.0
                if child.position in path_positions
                else -self._simulate(child, path_positions | {child.position})
            )
        node.visits += 1
        node.value_sum += value
        return value