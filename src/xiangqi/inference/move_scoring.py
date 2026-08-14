from __future__ import annotations

import torch
from torch import Tensor


MOVE_COUNT = 90 * 90


def joint_move_logits(start_logits: Tensor, end_logits: Tensor) -> Tensor:
    """Combine two 90-class heads into (batch, 8100) complete-move logits."""
    if start_logits.ndim != 2 or end_logits.ndim != 2:
        raise ValueError("move logits must have shape (batch, 90)")
    if start_logits.shape != end_logits.shape or start_logits.shape[1] != 90:
        raise ValueError("start/end logits must both have shape (batch, 90)")
    return (start_logits[:, :, None] + end_logits[:, None, :]).flatten(1)


def legal_move_mask(
    legal_moves: list[list[tuple[int, int]]],
    *,
    device: torch.device,
) -> Tensor:
    """Build a boolean (batch, 8100) mask from a rules engine's legal moves."""
    mask = torch.zeros((len(legal_moves), MOVE_COUNT), dtype=torch.bool, device=device)
    for batch_index, moves in enumerate(legal_moves):
        for start, end in moves:
            if not 0 <= start < 90 or not 0 <= end < 90:
                raise ValueError(f"invalid move index: {(start, end)}")
            mask[batch_index, start * 90 + end] = True
    return mask


def apply_legal_move_mask(move_logits: Tensor, mask: Tensor) -> Tensor:
    if move_logits.shape != mask.shape:
        raise ValueError("move logits and legal mask must have the same shape")
    if (~mask).all(dim=1).any():
        raise ValueError("a position has no legal moves")
    return move_logits.masked_fill(~mask, torch.finfo(move_logits.dtype).min)


def complete_move_topk(
    start_logits: Tensor,
    end_logits: Tensor,
    starts: Tensor,
    ends: Tensor,
    topk: tuple[int, ...] = (1, 5, 10),
    legal_moves: list[list[tuple[int, int]]] | None = None,
) -> dict[str, int | bool]:
    move_logits = joint_move_logits(start_logits, end_logits)
    if legal_moves is not None:
        move_logits = apply_legal_move_mask(
            move_logits,
            legal_move_mask(legal_moves, device=move_logits.device),
        )
    target = starts * 90 + ends
    candidates = move_logits.topk(max(topk), dim=1).indices
    result: dict[str, int | bool] = {
        f"complete_top{k}": int((candidates[:, :k] == target[:, None]).any(dim=1).sum())
        for k in topk
    }
    result["complete_masked"] = legal_moves is not None
    return result
