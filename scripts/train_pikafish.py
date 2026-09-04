from __future__ import annotations

import argparse
import json
import math
import random
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from training_models import PikafishResNet

CP_SCORE_KIND = 0
MATE_SCORE_KIND = 1
PROGRESS_LOG_INTERVAL = 20
J_SELECT_BASE_KL = 2.0510
J_SELECT_BASE_CP_MAE = 62.55
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.004
MATE_VALUE_WEIGHT = 4.0


@dataclass(frozen=True)
class ShardPaths:
    positions: Path
    teacher_score_kinds: Path
    teacher_scores: Path
    candidate_action_ids: Path
    candidate_score_kinds: Path
    candidate_scores: Path
    legal_action_ids: Path
    legal_action_offsets: Path
    length: int


def _shard_paths(data_dir: Path, split: str) -> list[ShardPaths]:
    shards: list[ShardPaths] = []
    for positions in sorted(data_dir.glob(f"{split}-*-positions.npy")):
        prefix = positions.name.removesuffix("-positions.npy")
        paths = {
            name: positions.with_name(f"{prefix}-{name}.npy")
            for name in (
                "teacher_score_kinds", "teacher_scores", "candidate_action_ids",
                "candidate_score_kinds", "candidate_scores", "legal_action_ids", "legal_action_offsets",
            )
        }
        missing = [path for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing shard arrays for {positions}: {missing}")
        length = len(np.load(positions, mmap_mode="r", allow_pickle=False))
        if len(np.load(paths["teacher_scores"], mmap_mode="r", allow_pickle=False)) != length:
            raise ValueError(f"teacher score length does not match {positions}")
        offsets = np.load(paths["legal_action_offsets"], mmap_mode="r", allow_pickle=False)
        if len(offsets) != length + 1:
            raise ValueError(f"legal action offsets do not match {positions}")
        shards.append(ShardPaths(positions=positions, length=length, **paths))
    return shards


class PikafishShardDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, data_dir: Path, split: str, cache_size: int = 16) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.shards = _shard_paths(data_dir, split)
        if not self.shards:
            raise FileNotFoundError(f"no {split} Pikafish shards under {data_dir}")
        self.ends = np.cumsum([shard.length for shard in self.shards], dtype=np.int64).tolist()
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.ends[-1]

    def _arrays(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        shard = self.shards[shard_index]
        arrays = {
            "positions": np.load(shard.positions, mmap_mode="r", allow_pickle=False),
            "teacher_score_kinds": np.load(shard.teacher_score_kinds, mmap_mode="r", allow_pickle=False),
            "teacher_scores": np.load(shard.teacher_scores, mmap_mode="r", allow_pickle=False),
            "candidate_action_ids": np.load(shard.candidate_action_ids, mmap_mode="r", allow_pickle=False),
            "candidate_score_kinds": np.load(shard.candidate_score_kinds, mmap_mode="r", allow_pickle=False),
            "candidate_scores": np.load(shard.candidate_scores, mmap_mode="r", allow_pickle=False),
            "legal_action_ids": np.load(shard.legal_action_ids, mmap_mode="r", allow_pickle=False),
            "legal_action_offsets": np.load(shard.legal_action_offsets, mmap_mode="r", allow_pickle=False),
        }
        self._cache[shard_index] = arrays
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        start = self.ends[shard_index - 1] if shard_index else 0
        local_index = index - start
        arrays = self._arrays(shard_index)
        legal_start = int(arrays["legal_action_offsets"][local_index])
        legal_end = int(arrays["legal_action_offsets"][local_index + 1])
        return {
            "positions": np.asarray(arrays["positions"][local_index]),
            "teacher_score_kinds": np.asarray(arrays["teacher_score_kinds"][local_index]),
            "teacher_scores": np.asarray(arrays["teacher_scores"][local_index]),
            "candidate_action_ids": np.asarray(arrays["candidate_action_ids"][local_index]),
            "candidate_score_kinds": np.asarray(arrays["candidate_score_kinds"][local_index]),
            "candidate_scores": np.asarray(arrays["candidate_scores"][local_index]),
            "legal_action_ids": np.asarray(arrays["legal_action_ids"][legal_start:legal_end]),
        }


class BlockShuffleSampler(Sampler[int]):
    def __init__(self, data_source: Dataset[Any], block_size: int = 65536, seed: int = 0) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.data_source = data_source
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = len(self.data_source)
        block_count = (total + self.block_size - 1) // self.block_size
        for block in torch.randperm(block_count, generator=generator).tolist():
            start = block * self.block_size
            size = min(self.block_size, total - start)
            for offset in torch.randperm(size, generator=generator).tolist():
                yield start + offset


def collate_pikafish(samples: list[dict[str, np.ndarray]]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    legal_offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    legal_offsets[1:] = np.cumsum([len(sample["legal_action_ids"]) for sample in samples], dtype=np.int64)
    return {
        "positions": torch.from_numpy(np.stack([sample["positions"] for sample in samples])),
        "teacher_score_kinds": torch.from_numpy(np.asarray([sample["teacher_score_kinds"] for sample in samples])),
        "teacher_scores": torch.from_numpy(np.asarray([sample["teacher_scores"] for sample in samples])),
        "candidate_action_ids": torch.from_numpy(np.stack([sample["candidate_action_ids"] for sample in samples])).long(),
        "candidate_score_kinds": torch.from_numpy(np.stack([sample["candidate_score_kinds"] for sample in samples])),
        "candidate_scores": torch.from_numpy(np.stack([sample["candidate_scores"] for sample in samples])),
        "legal_action_ids": torch.from_numpy(np.concatenate([sample["legal_action_ids"] for sample in samples])).long(),
        "legal_action_offsets": torch.from_numpy(legal_offsets),
    }


def _masked_log_probs(policy_logits: Tensor, legal_action_ids: Tensor, legal_action_offsets: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, action_count = policy_logits.shape
    legal_mask = torch.zeros((batch_size, action_count), dtype=torch.bool, device=policy_logits.device)
    legal_counts = (legal_action_offsets[1:] - legal_action_offsets[:-1]).to(torch.long)
    non_empty_rows = legal_counts > 0
    if non_empty_rows.any():
        row_ids = torch.repeat_interleave(torch.arange(batch_size, device=policy_logits.device), legal_counts)
        legal_mask[row_ids, legal_action_ids] = True
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    if (~non_empty_rows).any():
        masked_logits[~non_empty_rows, 0] = 0
    return torch.log_softmax(masked_logits, dim=1), legal_mask


def compute_losses(
    policy_logits: Tensor,
    value_predictions: Tensor,
    batch: dict[str, Tensor],
    *,
    temperature: float,
    value_scale: float,
) -> dict[str, Tensor]:
    if temperature <= 0 or value_scale <= 0:
        raise ValueError("temperature and value_scale must be positive")
    log_probs, legal_mask = _masked_log_probs(policy_logits, batch["legal_action_ids"], batch["legal_action_offsets"])
    actions = batch["candidate_action_ids"]
    candidate_mask = actions >= 0
    candidate_counts = candidate_mask.sum(dim=1)
    candidate_kinds = batch["candidate_score_kinds"]
    has_mate = ((candidate_kinds == MATE_SCORE_KIND) & candidate_mask).any(dim=1)
    has_only_cp = ((candidate_kinds == CP_SCORE_KIND) | ~candidate_mask).all(dim=1)
    policy_valid = (candidate_counts > 0) & (has_mate | has_only_cp)
    selected_rows, selected_cols = torch.nonzero(candidate_mask, as_tuple=True)
    if selected_rows.numel():
        selected_actions = actions[selected_rows, selected_cols]
        selected_legal = legal_mask[selected_rows, selected_actions]
        if (~selected_legal).any():
            invalid_rows = torch.unique(selected_rows[~selected_legal])
            policy_valid[invalid_rows] = False

    per_sample_policy = torch.zeros(policy_logits.shape[0], device=policy_logits.device)
    per_sample_cp_kl = torch.zeros(policy_logits.shape[0], device=policy_logits.device)
    cp_rows = policy_valid & ~has_mate
    if cp_rows.any():
        cp_actions = actions[cp_rows]
        cp_mask = candidate_mask[cp_rows]
        cp_scores = batch["candidate_scores"][cp_rows]
        cp_first_scores = cp_scores[:, :1]
        cp_target_logits = ((cp_scores - cp_first_scores) / temperature).masked_fill(~cp_mask, float("-inf"))
        cp_targets = torch.softmax(cp_target_logits, dim=1)
        gathered_log_probs = log_probs[cp_rows].gather(1, cp_actions.clamp_min(0)).masked_fill(~cp_mask, 0.0)
        cp_policy_loss = -(cp_targets * gathered_log_probs).sum(dim=1)
        cp_target_log = torch.where(cp_mask, cp_targets.clamp_min(1e-12).log(), torch.zeros_like(cp_targets))
        cp_kl = cp_policy_loss + (cp_targets * cp_target_log).sum(dim=1)
        per_sample_policy[cp_rows] = cp_policy_loss
        per_sample_cp_kl[cp_rows] = cp_kl
    mate_rows = policy_valid & has_mate
    if mate_rows.any():
        rows = torch.nonzero(mate_rows, as_tuple=False).flatten()
        per_sample_policy[rows] = -log_probs[rows, actions[rows, 0].clamp_min(0)]
    weights = torch.where(mate_rows, 4.0, 1.0).to(policy_logits.device) * policy_valid
    policy_loss = (per_sample_policy * weights).sum() / weights.sum().clamp_min(1)
    zero_policy_loss = per_sample_policy.sum() * 0
    policy_cp_loss = per_sample_policy[cp_rows].mean() if cp_rows.any() else zero_policy_loss
    policy_mate_loss = per_sample_policy[mate_rows].mean() if mate_rows.any() else zero_policy_loss

    teacher_scores = batch["teacher_scores"].to(value_predictions.dtype)
    is_cp = batch["teacher_score_kinds"] == CP_SCORE_KIND
    is_mate = batch["teacher_score_kinds"] == MATE_SCORE_KIND
    finite_scores = torch.isfinite(teacher_scores)
    mate_valid = is_mate & finite_scores & (teacher_scores != 0)
    value_valid = (is_cp & finite_scores) | mate_valid
    value_targets = torch.where(
        mate_valid,
        teacher_scores.sign(),
        torch.tanh(teacher_scores.clamp(-900, 900) / value_scale),
    )
    value_errors = torch.nn.functional.smooth_l1_loss(value_predictions.reshape(-1), value_targets, beta=0.1, reduction="none")
    value_weights = torch.where(mate_valid, MATE_VALUE_WEIGHT, 1.0).to(value_predictions.dtype)
    value_loss = (
        (value_errors * value_weights * value_valid).sum() / (value_weights * value_valid).sum().clamp_min(1)
        if value_valid.any() else value_errors.sum() * 0
    )
    value_cp_rows = is_cp & finite_scores
    zero_value_loss = value_errors.sum() * 0
    value_cp_loss = value_errors[value_cp_rows].mean() if value_cp_rows.any() else zero_value_loss
    value_mate_loss = value_errors[mate_valid].mean() if mate_valid.any() else zero_value_loss
    cp_policy_rows = cp_rows
    cp_policy_count = cp_policy_rows.sum()
    return {
        "policy_loss": policy_loss,
        "policy_cp_loss": policy_cp_loss,
        "policy_mate_loss": policy_mate_loss,
        "value_loss": value_loss,
        "value_cp_loss": value_cp_loss,
        "value_mate_loss": value_mate_loss,
        "policy_valid": policy_valid,
        "value_valid": value_valid,
        "value_mate": mate_valid,
        "mate_policy": mate_rows,
        "cp_policy_kl": per_sample_cp_kl[cp_policy_rows].mean() if cp_policy_count else per_sample_cp_kl.sum() * 0,
        "cp_policy_count": cp_policy_count,
        "mate_policy_count": mate_rows.sum(),
        "value_cp_count": value_cp_rows.sum(),
        "value_mate_count": mate_valid.sum(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def mirror_action_ids(action_ids: Tensor) -> Tensor:
    valid = action_ids >= 0
    squares = action_ids.clamp_min(0)
    starts, ends = squares.div(90, rounding_mode="floor"), squares.remainder(90)
    mirrored_starts = (starts // 9) * 9 + (8 - starts.remainder(9))
    mirrored_ends = (ends // 9) * 9 + (8 - ends.remainder(9))
    return torch.where(valid, mirrored_starts * 90 + mirrored_ends, action_ids)


def with_horizontal_mirror(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    positions = batch["positions"]
    legal_action_ids = batch["legal_action_ids"]
    legal_action_offsets = batch["legal_action_offsets"]
    mirrored_legal_action_ids = mirror_action_ids(legal_action_ids)
    legal_count = len(legal_action_ids)
    return {
        **{
            name: torch.cat((value, value), dim=0)
            for name, value in batch.items()
            if name not in {"positions", "candidate_action_ids", "legal_action_ids", "legal_action_offsets"}
        },
        "positions": torch.cat((positions, torch.flip(positions, dims=[3])), dim=0),
        "candidate_action_ids": torch.cat((
            batch["candidate_action_ids"], mirror_action_ids(batch["candidate_action_ids"]),
        ), dim=0),
        "legal_action_ids": torch.cat((legal_action_ids, mirrored_legal_action_ids)),
        "legal_action_offsets": torch.cat((
            legal_action_offsets,
            legal_action_offsets[1:] + legal_count,
        )),
    }


def joint_loss(losses: dict[str, Tensor], policy_weight: float, value_weight: float) -> Tensor:
    return policy_weight * losses["policy_loss"] + value_weight * losses["value_loss"]


def compute_j_select(validation: dict[str, float]) -> float:
    return (
        0.4 * (validation["cp_policy_kl"] / J_SELECT_BASE_KL)
        + 0.4 * (validation["value_cp_mae_le_300"] / J_SELECT_BASE_CP_MAE)
        + 0.2 * (1 - validation["value_sign_accuracy"])
    )


def save_best_checkpoint(checkpoint: dict[str, object], checkpoint_dir: Path, name: str) -> None:
    torch.save(checkpoint, checkpoint_dir / name)


def evaluate(
    model: PikafishResNet,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    temperature: float,
    value_scale: float,
    policy_weight: float,
    value_weight: float,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    totals = {"joint_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    policy_valid = value_valid = mate_policy = value_mate = samples = cp_policy_samples = 0
    policy_cp_loss_total = policy_mate_loss_total = value_cp_loss_total = value_mate_loss_total = 0.0
    policy_mate_samples = value_cp_samples = value_mate_samples = 0
    cp_mae_total = cp_mae_le_100_total = cp_mae_all_total = cp_mae_gt_300_total = 0.0
    cp_mae_samples = cp_mae_le_100_samples = cp_mae_all_samples = cp_mae_gt_300_samples = 0
    sign_correct = sign_samples = sign_le_100_correct = sign_le_100_samples = sign_gt_300_correct = sign_gt_300_samples = 0
    value_target_mae_total = value_target_mae_samples = 0.0
    cp_policy_kl_total = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                policy_logits, value_predictions = model(batch["positions"])
                losses = compute_losses(
                    policy_logits, value_predictions, batch,
                    temperature=temperature, value_scale=value_scale,
                )
                loss = joint_loss(losses, policy_weight, value_weight)
            batch_size = len(batch["positions"])
            samples += batch_size
            for name in totals:
                totals[name] += float(losses[name] if name != "joint_loss" else loss) * batch_size
            policy_valid += int(losses["policy_valid"].sum())
            value_valid += int(losses["value_valid"].sum())
            mate_policy += int(losses["mate_policy"].sum())
            value_mate += int(losses["value_mate"].sum())
            cp_policy_count = int(losses["cp_policy_count"])
            cp_policy_kl_total += float(losses["cp_policy_kl"]) * cp_policy_count
            cp_policy_samples += cp_policy_count
            mate_policy_count = int(losses["mate_policy_count"])
            value_cp_count = int(losses["value_cp_count"])
            value_mate_count = int(losses["value_mate_count"])
            policy_cp_loss_total += float(losses["policy_cp_loss"]) * cp_policy_count
            policy_mate_loss_total += float(losses["policy_mate_loss"]) * mate_policy_count
            value_cp_loss_total += float(losses["value_cp_loss"]) * value_cp_count
            value_mate_loss_total += float(losses["value_mate_loss"]) * value_mate_count
            policy_mate_samples += mate_policy_count
            value_cp_samples += value_cp_count
            value_mate_samples += value_mate_count
            teacher_scores = batch["teacher_scores"]
            value_targets = torch.where(
                losses["value_mate"],
                teacher_scores.sign(),
                torch.tanh(teacher_scores.clamp(-900, 900) / value_scale),
            )
            value_target_mae_total += torch.abs(value_predictions.reshape(-1) - value_targets)[losses["value_valid"]].sum().item()
            value_target_mae_samples += int(losses["value_valid"].sum())
            cp_mask = (batch["teacher_score_kinds"] == CP_SCORE_KIND) & torch.isfinite(teacher_scores)
            predicted_cp = value_scale * torch.atanh(value_predictions.reshape(-1).float().clamp(-0.999, 0.999))
            absolute_error = torch.abs(predicted_cp - teacher_scores.float())
            cp_mae_mask = cp_mask & (teacher_scores.abs() <= 300)
            cp_mae_le_100_mask = cp_mask & (teacher_scores.abs() <= 100)
            cp_mae_gt_300_mask = cp_mask & (teacher_scores.abs() > 300)
            for mask, total_name in (
                (cp_mae_mask, "low"),
                (cp_mae_le_100_mask, "balanced"),
                (cp_mask, "all"),
                (cp_mae_gt_300_mask, "high"),
            ):
                if mask.any():
                    error_sum = absolute_error[mask].sum().item()
                    count = int(mask.sum())
                    if total_name == "low":
                        cp_mae_total += error_sum
                        cp_mae_samples += count
                    elif total_name == "balanced":
                        cp_mae_le_100_total += error_sum
                        cp_mae_le_100_samples += count
                    elif total_name == "all":
                        cp_mae_all_total += error_sum
                        cp_mae_all_samples += count
                    else:
                        cp_mae_gt_300_total += error_sum
                        cp_mae_gt_300_samples += count
            sign_mask = cp_mask & (teacher_scores != 0)
            if sign_mask.any():
                sign_correct += int((value_predictions.reshape(-1)[sign_mask].sign() == teacher_scores[sign_mask].sign()).sum())
                sign_samples += int(sign_mask.sum())
            sign_le_100_mask = sign_mask & (teacher_scores.abs() <= 100)
            if sign_le_100_mask.any():
                sign_le_100_correct += int((value_predictions.reshape(-1)[sign_le_100_mask].sign() == teacher_scores[sign_le_100_mask].sign()).sum())
                sign_le_100_samples += int(sign_le_100_mask.sum())
            sign_gt_300_mask = sign_mask & (teacher_scores.abs() > 300)
            if sign_gt_300_mask.any():
                sign_gt_300_correct += int((value_predictions.reshape(-1)[sign_gt_300_mask].sign() == teacher_scores[sign_gt_300_mask].sign()).sum())
                sign_gt_300_samples += int(sign_gt_300_mask.sum())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    denominator = max(samples, 1)
    return {
        **{name: value / denominator for name, value in totals.items()},
        "samples": float(samples),
        "policy_valid": float(policy_valid),
        "value_valid": float(value_valid),
        "value_mate": float(value_mate),
        "mate_policy": float(mate_policy),
        "policy_cp_loss": policy_cp_loss_total / max(cp_policy_samples, 1),
        "policy_mate_loss": policy_mate_loss_total / max(policy_mate_samples, 1),
        "policy_mate_samples": float(policy_mate_samples),
        "value_cp_loss": value_cp_loss_total / max(value_cp_samples, 1),
        "value_mate_loss": value_mate_loss_total / max(value_mate_samples, 1),
        "value_cp_samples": float(value_cp_samples),
        "value_mate_samples": float(value_mate_samples),
        "value_target_mae": value_target_mae_total / max(value_target_mae_samples, 1),
        "value_cp_mae_le_300": cp_mae_total / max(cp_mae_samples, 1),
        "value_cp_mae_samples": float(cp_mae_samples),
        "value_cp_mae_le_100": cp_mae_le_100_total / max(cp_mae_le_100_samples, 1),
        "value_cp_mae_le_100_samples": float(cp_mae_le_100_samples),
        "value_cp_mae_all": cp_mae_all_total / max(cp_mae_all_samples, 1),
        "value_cp_mae_all_samples": float(cp_mae_all_samples),
        "value_cp_mae_gt_300": cp_mae_gt_300_total / max(cp_mae_gt_300_samples, 1),
        "value_cp_mae_gt_300_samples": float(cp_mae_gt_300_samples),
        "value_sign_accuracy": sign_correct / max(sign_samples, 1),
        "value_sign_samples": float(sign_samples),
        "value_sign_accuracy_le_100": sign_le_100_correct / max(sign_le_100_samples, 1),
        "value_sign_le_100_samples": float(sign_le_100_samples),
        "value_sign_accuracy_gt_300": sign_gt_300_correct / max(sign_gt_300_samples, 1),
        "value_sign_gt_300_samples": float(sign_gt_300_samples),
        "cp_policy_kl": cp_policy_kl_total / max(cp_policy_samples, 1),
        "cp_policy_samples": float(cp_policy_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Pikafish joint-policy distillation model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/pikafish-distillation/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pikafish-c192-b12"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--value-scale", type=float, default=300.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mirror", action="store_true", help="append horizontally mirrored training positions")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-ratio", type=float, default=0.05)
    warmup.add_argument("--warmup-steps", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.global_batch_size < args.micro_batch_size or args.global_batch_size % args.micro_batch_size:
        parser.error("--global-batch-size must be a positive multiple of --micro-batch-size")
    if args.epochs < 1 or args.temperature <= 0 or args.value_scale <= 0:
        parser.error("epochs, temperature, and value scale must be positive")
    if args.learning_rate <= 0 or args.value_learning_rate <= 0 or args.min_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate must be <= --learning-rate")
    if args.policy_weight < 0 or args.value_weight < 0:
        parser.error("loss weights must be non-negative")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.warmup_steps is not None and args.warmup_steps < 1:
        parser.error("--warmup-steps must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("num-workers must be non-negative and prefetch-factor must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_dataset = PikafishShardDataset(args.data_dir, "train")
    validation_dataset = PikafishShardDataset(args.data_dir, "validation")
    sampler = BlockShuffleSampler(train_dataset, block_size=args.block_size, seed=args.seed)
    loader_options: dict[str, object] = {
        "batch_size": args.micro_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_pikafish,
    }
    if args.num_workers:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = PikafishResNet().to(device)
    value_parameter_ids = {id(parameter) for parameter in model.value_head.parameters()}
    main_parameters = [parameter for parameter in model.parameters() if id(parameter) not in value_parameter_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": args.learning_rate},
            {"params": model.value_head.parameters(), "lr": args.value_learning_rate},
        ],
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    accumulation_steps = args.global_batch_size // args.micro_batch_size
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = args.epochs * updates_per_epoch
    target_steps = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, math.ceil(total_steps * args.warmup_ratio))
    if warmup_steps >= total_steps:
        parser.error("warmup steps must be less than total training steps")

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return (args.min_learning_rate / args.learning_rate) + (1 - args.min_learning_rate / args.learning_rate) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (args.checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8")
    progress_log_file = (args.checkpoint_dir / "progress.jsonl").open("a", encoding="utf-8")

    def emit_log(payload: dict[str, object], *, target: Any = None) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        print(message, flush=True)
        output = progress_log_file if target is None else target
        output.write(f"{message}\n")
        output.flush()

    def make_checkpoint(
        *,
        epoch_completed: int,
        next_epoch_index: int,
        next_batch_index: int,
        epoch_update_step: int,
        validation: dict[str, float] | None,
    ) -> dict[str, object]:
        validation_with_select = None
        if validation is not None:
            validation_with_select = {**validation, "j_select": compute_j_select(validation)}
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch_completed,
            "global_step": global_step,
            "validation": validation_with_select,
            "best_j_select": best_j_select,
            "best_policy_kl": best_policy_kl,
            "best_value_cp_mae": best_value_cp_mae,
            "schedule_total_steps": total_steps,
            "schedule_warmup_steps": warmup_steps,
            "config": vars(args),
            "training_state": {
                "next_epoch_index": next_epoch_index,
                "next_batch_index": next_batch_index,
                "epoch_update_step": epoch_update_step,
                "early_stopping_j_select": early_stopping_j_select,
                "no_improve_epochs": no_improve_epochs,
            },
        }

    best_j_select = float("inf")
    best_policy_kl = float("inf")
    best_value_cp_mae = float("inf")
    early_stopping_j_select = float("inf")
    no_improve_epochs = 0
    global_step = 0
    train_started = time.perf_counter()
    last_update_time = train_started
    average_update_seconds = 0.0
    timed_updates = 0
    last_gradient_norm = 0.0
    start_epoch_index = 0
    start_batch_index = 0
    start_epoch_update_step = 0

    resume_path = args.checkpoint_dir / "last.pt"
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        saved_schedule_total_steps = checkpoint.get("schedule_total_steps")
        saved_schedule_warmup_steps = checkpoint.get("schedule_warmup_steps")
        if saved_schedule_total_steps is None:
            saved_config = checkpoint.get("config") or {}
            saved_micro_batch_size = int(saved_config.get("micro_batch_size", args.micro_batch_size))
            saved_global_batch_size = int(saved_config.get("global_batch_size", args.global_batch_size))
            saved_accumulation_steps = saved_global_batch_size // saved_micro_batch_size
            saved_updates_per_epoch = math.ceil(
                math.ceil(len(train_dataset) / saved_micro_batch_size) / saved_accumulation_steps
            )
            saved_schedule_total_steps = int(saved_config.get("epochs", args.epochs)) * saved_updates_per_epoch
            saved_schedule_warmup_steps = int(saved_config.get("warmup_steps") or max(
                1, math.ceil(saved_schedule_total_steps * float(saved_config.get("warmup_ratio", 0.05)))
            ))
        if (
            saved_schedule_total_steps != total_steps
            or saved_schedule_warmup_steps != warmup_steps
        ):
            raise ValueError(
                "resume schedule differs from checkpoint; keep --epochs and warmup settings unchanged "
                "to preserve learning-rate continuity"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        global_step = int(checkpoint.get("global_step", 0))
        best_j_select = float(checkpoint.get("best_j_select", checkpoint.get("best_validation_loss", float("inf"))))
        checkpoint_validation = checkpoint.get("validation") or {}
        best_policy_kl = float(checkpoint.get("best_policy_kl", checkpoint_validation.get("cp_policy_kl", float("inf"))))
        best_value_cp_mae = float(checkpoint.get("best_value_cp_mae", checkpoint_validation.get("value_cp_mae_le_300", float("inf"))))
        training_state = checkpoint.get("training_state") or {}
        early_stopping_j_select = float(training_state.get("early_stopping_j_select", best_j_select))
        start_epoch_index = int(training_state.get("next_epoch_index", int(checkpoint.get("epoch", 0))))
        start_batch_index = int(training_state.get("next_batch_index", 0))
        start_epoch_update_step = int(training_state.get("epoch_update_step", 0))
        no_improve_epochs = int(training_state.get("no_improve_epochs", 0))
        emit_log({
            "event": "resume",
            "resume_from": str(resume_path),
            "start_epoch": start_epoch_index + 1,
            "start_batch_index": start_batch_index,
            "global_step": global_step,
            "best_j_select": best_j_select,
            "best_policy_kl": best_policy_kl,
            "best_value_cp_mae": best_value_cp_mae,
            "early_stopping_j_select": early_stopping_j_select,
            "no_improve_epochs": no_improve_epochs,
        })

    logged_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    emit_log({
        "event": "training_start",
        "device": str(device), "train_samples": len(train_dataset), "validation_samples": len(validation_dataset),
        "micro_batch_size": args.micro_batch_size, "global_batch_size": args.global_batch_size,
        "accumulation_steps": accumulation_steps, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_steps": target_steps, "warmup_steps": warmup_steps, "config": logged_config,
    })

    try:
        for epoch in range(start_epoch_index, args.epochs):
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            epoch_update_step = start_epoch_update_step if epoch == start_epoch_index else 0
            interrupted_mid_epoch = False
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch_index and batch_index < start_batch_index:
                    continue
                batch = move_batch(batch, device)
                if args.mirror:
                    batch = with_horizontal_mirror(batch)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    policy_logits, value_predictions = model(batch["positions"])
                    losses = compute_losses(
                        policy_logits, value_predictions, batch,
                        temperature=args.temperature, value_scale=args.value_scale,
                    )
                    loss = joint_loss(losses, args.policy_weight, args.value_weight)
                scaler.scale(loss / accumulation_steps).backward()
                should_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
                if should_update:
                    scaler.unscale_(optimizer)
                    last_gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                    gradient_was_clipped = last_gradient_norm > args.max_grad_norm
                    amp_scale_before_update = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_applied = not amp_enabled or scaler.get_scale() >= amp_scale_before_update
                    optimizer.zero_grad(set_to_none=True)
                    if not optimizer_step_applied:
                        continue
                    scheduler.step()
                    global_step += 1
                    epoch_update_step += 1
                    now = time.perf_counter()
                    update_seconds = now - last_update_time
                    last_update_time = now
                    timed_updates += 1
                    average_update_seconds += (update_seconds - average_update_seconds) / timed_updates
                    should_log = (
                        global_step == 1
                        or global_step % PROGRESS_LOG_INTERVAL == 0
                        or global_step >= target_steps
                        or batch_index + 1 == len(train_loader)
                    )
                    if should_log:
                        epoch_elapsed_seconds = now - started
                        epoch_remaining_updates = max(updates_per_epoch - epoch_update_step, 0)
                        value_outputs = value_predictions.detach().float().reshape(-1)
                        emit_log({
                            "event": "train_progress",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "epoch_progress": epoch_update_step / updates_per_epoch,
                            "epoch_elapsed_seconds": epoch_elapsed_seconds,
                            "epoch_eta_seconds": epoch_remaining_updates * average_update_seconds,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "value_learning_rate": optimizer.param_groups[1]["lr"],
                            "joint_loss": float(loss),
                            "policy_loss": float(losses["policy_loss"]),
                            "policy_cp_loss": float(losses["policy_cp_loss"]),
                            "policy_mate_loss": float(losses["policy_mate_loss"]),
                            "value_loss": float(losses["value_loss"]),
                            "value_cp_loss": float(losses["value_cp_loss"]),
                            "value_mate_loss": float(losses["value_mate_loss"]),
                            "cp_policy_kl": float(losses["cp_policy_kl"]),
                            "policy_valid_count": int(losses["policy_valid"].sum()),
                            "value_valid_count": int(losses["value_valid"].sum()),
                            "mate_policy_count": int(losses["mate_policy_count"]),
                            "value_mate_count": int(losses["value_mate_count"]),
                            "value_prediction_mean": float(value_outputs.mean()),
                            "value_saturation_ratio": float((value_outputs.abs() >= 0.99).float().mean()),
                            "gradient_norm_pre_clip": last_gradient_norm,
                            "gradient_clipped": gradient_was_clipped,
                        }, target=progress_log_file)
                if args.max_steps is not None and global_step >= args.max_steps:
                    interrupted_mid_epoch = batch_index + 1 < len(train_loader)
                    break

            if interrupted_mid_epoch:
                break

            validation = evaluate(
                model, validation_loader, device,
                temperature=args.temperature, value_scale=args.value_scale,
                policy_weight=args.policy_weight, value_weight=args.value_weight,
                amp_enabled=amp_enabled, max_batches=50 if args.max_steps else None,
            )
            validation_j_select = compute_j_select(validation)
            is_best = validation_j_select < best_j_select
            is_best_policy = validation["cp_policy_kl"] < best_policy_kl
            is_best_value = validation["value_cp_mae_le_300"] < best_value_cp_mae
            is_significant_improvement = validation_j_select < early_stopping_j_select - EARLY_STOPPING_MIN_DELTA
            if is_best:
                best_j_select = validation_j_select
            if is_best_policy:
                best_policy_kl = validation["cp_policy_kl"]
            if is_best_value:
                best_value_cp_mae = validation["value_cp_mae_le_300"]
            if is_significant_improvement:
                early_stopping_j_select = validation_j_select
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            checkpoint = make_checkpoint(
                epoch_completed=epoch + 1,
                next_epoch_index=epoch + 1,
                next_batch_index=0,
                epoch_update_step=0,
                validation=validation,
            )
            torch.save(checkpoint, args.checkpoint_dir / "last.pt")
            if is_best:
                save_best_checkpoint(checkpoint, args.checkpoint_dir, "best.pt")
            if is_best_policy:
                save_best_checkpoint(checkpoint, args.checkpoint_dir, "best-policy.pt")
            if is_best_value:
                save_best_checkpoint(checkpoint, args.checkpoint_dir, "best-value.pt")
            emit_log({
                "epoch": epoch + 1, "global_step": global_step, "validation": validation,
                "validation_j_select": validation_j_select,
                "best_j_select": best_j_select,
                "best_policy_kl": best_policy_kl,
                "best_value_cp_mae": best_value_cp_mae,
                "early_stopping_j_select": early_stopping_j_select,
                "no_improve_epochs": no_improve_epochs,
                "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - started,
                "value_learning_rate": optimizer.param_groups[1]["lr"],
                "gradient_norm_pre_clip": last_gradient_norm,
            }, target=metrics_file)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                emit_log({
                    "event": "early_stop",
                    "epoch": epoch + 1,
                    "no_improve_epochs": no_improve_epochs,
                    "patience": EARLY_STOPPING_PATIENCE,
                    "best_j_select": best_j_select,
                })
                break
            start_batch_index = 0
            start_epoch_update_step = 0
    finally:
        metrics_file.close()
        progress_log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())