from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from backend.models import ResNet
from backend.inference import complete_move_topk


@dataclass(frozen=True)
class ShardPaths:
    positions: Path
    start_indices: Path
    end_indices: Path
    values: Path | None = None

def find_shards(data_dir: Path, split: str) -> list[ShardPaths]:
    position_paths = sorted(data_dir.glob(f"{split}-*-positions.npy"))
    shards = []
    for positions in position_paths:
        prefix = positions.name.removesuffix("-positions.npy")
        starts = positions.with_name(f"{prefix}-start_indices.npy")
        ends = positions.with_name(f"{prefix}-end_indices.npy")
        if not starts.exists() or not ends.exists():
            raise FileNotFoundError(f"missing label arrays for {positions}")
        values = positions.with_name(f"{prefix}-values.npy")
        shards.append(ShardPaths(positions, starts, ends, values if values.exists() else None))
    return shards


class ShardDataset(IterableDataset):
    def __init__(self, shards: list[ShardPaths], batch_size: int, worker_count: int = 1, value_head: bool = False) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self.shards = shards
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.value_head = value_head
        self.sizes: list[int] = []
        self.training = False
        for shard in shards:
            size = len(np.load(shard.positions, mmap_mode="r"))
            if len(np.load(shard.start_indices, mmap_mode="r")) != size:
                raise ValueError(f"label length does not match positions in {shard.positions}")
            if self.value_head:
                if shard.values is None:
                    raise FileNotFoundError(f"value labels required for {shard.positions}")
                if len(np.load(shard.values, mmap_mode="r")) != size:
                    raise ValueError(f"value label length does not match positions in {shard.positions}")
            self.sizes.append(size)

    def __len__(self) -> int:
        return sum(
            (
                sum(self.sizes[worker_id::self.worker_count])
                + self.batch_size
                - 1
            )
            // self.batch_size
            for worker_id in range(self.worker_count)
        )

    def sample_count(self) -> int:
        return sum(self.sizes)

    def _load_shard(self, shard: ShardPaths) -> tuple[np.ndarray, ...]:
        arrays: tuple[np.ndarray, ...] = (
            np.load(shard.positions, mmap_mode="r"),
            np.load(shard.start_indices, mmap_mode="r"),
            np.load(shard.end_indices, mmap_mode="r"),
        )
        if self.value_head:
            assert shard.values is not None
            arrays += (np.load(shard.values, mmap_mode="r"),)
        return arrays

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        shards = self.shards[worker_id::worker_count]
        rng = np.random.default_rng(torch.initial_seed() + worker_id)
        pending: tuple[np.ndarray, ...] | None = None
        if self.training:
            shards = list(shards)
            rng.shuffle(shards)
        for shard in shards:
            arrays = self._load_shard(shard)
            positions, starts, ends = arrays[:3]
            indices = np.arange(len(positions))
            if self.training:
                rng.shuffle(indices)
            offset = 0
            if pending is not None:
                required = self.batch_size - len(pending[0])
                batch_indices = indices[:required]
                current: tuple[np.ndarray, ...] = (
                    np.asarray(positions[batch_indices]),
                    np.asarray(starts[batch_indices]),
                    np.asarray(ends[batch_indices]),
                )
                if self.value_head:
                    current += (np.asarray(arrays[3][batch_indices]),)
                if len(current[0]) == required:
                    yield tuple(
                        torch.from_numpy(np.concatenate((previous, addition)))
                        for previous, addition in zip(pending, current)
                    )
                    pending = None
                    offset = required
                else:
                    pending = tuple(
                        np.concatenate((previous, addition))
                        for previous, addition in zip(pending, current)
                    )
                    continue
            while offset + self.batch_size <= len(indices):
                batch_indices = indices[offset : offset + self.batch_size]
                batch: tuple[torch.Tensor, ...] = (
                    torch.from_numpy(np.asarray(positions[batch_indices])),
                    torch.from_numpy(np.asarray(starts[batch_indices])),
                    torch.from_numpy(np.asarray(ends[batch_indices])),
                )
                if self.value_head:
                    batch += (torch.from_numpy(np.asarray(arrays[3][batch_indices])),)
                yield batch
                offset += self.batch_size
            if offset < len(indices):
                batch_indices = indices[offset:]
                pending = (
                    np.asarray(positions[batch_indices]),
                    np.asarray(starts[batch_indices]),
                    np.asarray(ends[batch_indices]),
                )
                if self.value_head:
                    pending += (np.asarray(arrays[3][batch_indices]),)
        if pending is not None:
            yield tuple(torch.from_numpy(values) for values in pending)

    def set_training(self, training: bool) -> None:
        self.training = training


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_metrics(path: Path, metrics: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit: int | None,
    value_head: bool = False,
    value_weight: float = 0.5,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_value_absolute_error = 0.0
    value_sum = value_squared_sum = prediction_sum = prediction_squared_sum = prediction_value_sum = 0.0
    total_samples = 0
    start_top1 = start_top5 = end_top1 = end_top5 = 0
    complete_top1 = complete_top5 = complete_top10 = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            positions, starts, ends = batch[:3]
            positions = positions.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            ends = ends.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(positions)
            start_logits, end_logits = outputs[:2]
            policy_loss = criterion(start_logits, starts) + criterion(end_logits, ends)
            loss = policy_loss
            if value_head:
                values = batch[3].to(device, non_blocking=True)
                predictions = outputs[2].reshape(-1)
                values = values.reshape(-1)
                value_loss = nn.functional.mse_loss(predictions, values)
                loss = loss + value_weight * value_loss
                total_value_loss += value_loss.item() * len(positions)
                total_value_absolute_error += torch.abs(predictions - values).sum().item()
                value_sum += values.sum().item()
                value_squared_sum += torch.square(values).sum().item()
                prediction_sum += predictions.sum().item()
                prediction_squared_sum += torch.square(predictions).sum().item()
                prediction_value_sum += (predictions * values).sum().item()
            total_loss += loss.item() * len(positions)
            total_policy_loss += policy_loss.item() * len(positions)
            total_samples += len(positions)
            start_top1 += (start_logits.argmax(1) == starts).sum().item()
            end_top1 += (end_logits.argmax(1) == ends).sum().item()
            start_top5 += (start_logits.topk(5, dim=1).indices == starts[:, None]).any(1).sum().item()
            end_top5 += (end_logits.topk(5, dim=1).indices == ends[:, None]).any(1).sum().item()
            complete = complete_move_topk(start_logits, end_logits, starts, ends)
            complete_top1 += complete["complete_top1"]
            complete_top5 += complete["complete_top5"]
            complete_top10 += complete["complete_top10"]
            if limit is not None and batch_index + 1 >= limit:
                break
    denominator = max(total_samples, 1)
    metrics = {
        "loss": total_loss / denominator,
        "policy_loss": total_policy_loss / denominator,
        "start_top1": start_top1 / denominator,
        "start_top5": start_top5 / denominator,
        "end_top1": end_top1 / denominator,
        "end_top5": end_top5 / denominator,
        "complete_top1": complete_top1 / denominator,
        "complete_top5": complete_top5 / denominator,
        "complete_top10": complete_top10 / denominator,
        "complete_masked": False,
    }
    if value_head:
        value_mean = value_sum / denominator
        prediction_mean = prediction_sum / denominator
        covariance = prediction_value_sum / denominator - prediction_mean * value_mean
        value_variance = max(value_squared_sum / denominator - value_mean * value_mean, 0.0)
        prediction_variance = max(prediction_squared_sum / denominator - prediction_mean * prediction_mean, 0.0)
        denominator_correlation = (value_variance * prediction_variance) ** 0.5
        metrics.update({
            "value_loss": total_value_loss / denominator,
            "value_mae": total_value_absolute_error / denominator,
            "value_correlation": covariance / denominator_correlation if denominator_correlation > 0 else 0.0,
        })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Xiangqi ResNet policy baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--current-view", action="store_true", help="train with current-side-view inputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4, help="batches prefetched per DataLoader worker")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="use CUDA mixed precision")
    parser.add_argument("--value-head", action="store_true", help="train a new model with a supervised value head")
    parser.add_argument("--value-weight", type=float, default=0.5, help="weight of value loss when using --value-head")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="CHECKPOINT",
        help="resume from checkpoint-dir/last.pt, or from CHECKPOINT when provided",
    )
    args = parser.parse_args()
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")
    if args.value_weight < 0:
        parser.error("--value-weight must be non-negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train_shards = find_shards(args.data_dir, "train")
    validation_shards = find_shards(args.data_dir, "validation")
    test_shards = find_shards(args.data_dir, "test")
    if not train_shards or not validation_shards or not test_shards:
        raise FileNotFoundError(f"train/validation/test shards not found under {args.data_dir}")

    worker_count = max(args.num_workers, 1)
    train_dataset = ShardDataset(train_shards, args.batch_size, worker_count, args.value_head)
    validation_dataset = ShardDataset(validation_shards, args.batch_size, worker_count, args.value_head)
    train_dataset.set_training(True)
    loader_options = {
        "batch_size": None,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    model = ResNet(
        channels=args.channels,
        blocks=args.blocks,
        value_head=args.value_head,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        fused=device.type == "cuda",
    )
    updates_per_epoch = max(1, (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * updates_per_epoch
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    start_epoch = 0

    if args.resume:
        resume_path = args.checkpoint_dir / "last.pt" if args.resume is True else Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", checkpoint["validation_loss"]))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(json.dumps({
            "resumed_from": str(resume_path),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
        }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "device": str(device),
        "train_samples": train_dataset.sample_count(),
        "validation_samples": validation_dataset.sample_count(),
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "amp": amp_enabled,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, ensure_ascii=False))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        epoch_batches = len(train_loader)
        training_loss_total = 0.0
        training_policy_loss_total = 0.0
        training_value_loss_total = 0.0
        training_samples = 0
        for batch_index, batch in enumerate(train_loader):
            positions, starts, ends = batch[:3]
            positions = positions.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            ends = ends.to(device, non_blocking=True)
            if args.mirror:
                mirrored_positions = torch.flip(positions, dims=[3])
                width = positions.shape[3]
                mirrored_starts = (starts // width) * width + (width - 1 - starts % width)
                mirrored_ends = (ends // width) * width + (width - 1 - ends % width)
                positions = torch.cat((positions, mirrored_positions), dim=0)
                starts = torch.cat((starts, mirrored_starts), dim=0)
                ends = torch.cat((ends, mirrored_ends), dim=0)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(positions)
                start_logits, end_logits = outputs[:2]
                policy_loss = criterion(start_logits, starts) + criterion(end_logits, ends)
                batch_loss = policy_loss
                value_loss = None
                if args.value_head:
                    values = batch[3].to(device, non_blocking=True)
                    if args.mirror:
                        values = torch.cat((values, values), dim=0)
                    value_loss = nn.functional.mse_loss(outputs[2].reshape(-1), values.reshape(-1))
                    batch_loss = batch_loss + args.value_weight * value_loss
            training_loss_total += batch_loss.item() * len(positions)
            training_policy_loss_total += policy_loss.item() * len(positions)
            if value_loss is not None:
                training_value_loss_total += value_loss.item() * len(positions)
            training_samples += len(positions)
            loss = batch_loss / args.accumulation_steps
            scaler.scale(loss).backward()
            should_update = (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_applied = scaler.get_scale() >= previous_scale
                if optimizer_step_applied:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
            completed_batches = batch_index + 1
            if completed_batches == 1 or completed_batches % 100 == 0 or completed_batches == epoch_batches:
                elapsed = time.perf_counter() - epoch_started
                batches_per_second = completed_batches / elapsed if elapsed else 0.0
                remaining = epoch_batches - completed_batches
                eta_seconds = remaining / batches_per_second if batches_per_second else 0.0
                memory = ""
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                    memory = (
                        f" gpu_mem={allocated:.2f}GiB"
                        f" reserved={reserved:.2f}GiB"
                        f" max={max_allocated:.2f}GiB"
                    )
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"batch={completed_batches}/{epoch_batches} "
                    f"loss={loss.item() * args.accumulation_steps:.5f} "
                    f"step={global_step} "
                    f"speed={batches_per_second:.2f}batch/s "
                    f"eta={eta_seconds / 60:.1f}min{memory}",
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            limit=50 if args.max_steps else None,
            value_head=args.value_head,
            value_weight=args.value_weight,
            amp_enabled=amp_enabled,
        )
        validation_loss = validation_metrics["loss"]
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": vars(args),
        }
        torch.save(checkpoint, args.checkpoint_dir / "last.pt")
        torch.save(checkpoint, args.checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")
        if improved:
            torch.save(checkpoint, args.checkpoint_dir / "best.pt")
        append_metrics(
            args.checkpoint_dir / "metrics.jsonl",
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "training_loss": training_loss_total / max(training_samples, 1),
                "training_policy_loss": training_policy_loss_total / max(training_samples, 1),
                **({"training_value_loss": training_value_loss_total / max(training_samples, 1)} if args.value_head else {}),
                "validation": validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            },
        )
        print(json.dumps({"epoch": epoch + 1, "validation": validation_metrics, "step": global_step}, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement", flush=True)
            break
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    test_dataset = ShardDataset(test_shards, args.batch_size, worker_count, args.value_head)
    test_loader = DataLoader(
        test_dataset,
        **loader_options,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        limit=50 if args.max_steps else None,
        value_head=args.value_head,
        value_weight=args.value_weight,
        amp_enabled=amp_enabled,
    )
    print(json.dumps({"test": test_metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
