from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from xiangqi.models import TinyResNet
from xiangqi.inference import complete_move_topk


class ShardDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[Path], index_cache: Path | None = None) -> None:
        self.paths = paths
        self.sizes: list[int] = []
        self.training = False
        self.index_cache = index_cache
        cached = self._read_index_cache()
        changed = False
        for path in paths:
            key = str(path.resolve())
            signature = self._signature(path)
            size = cached.get(key, {}).get("size")
            if cached.get(key, {}).get("signature") != signature or size is None:
                with np.load(path) as data:
                    size = len(data["positions"])
                cached[key] = {"signature": signature, "size": size}
                changed = True
            self.sizes.append(int(size))
        if changed:
            self._write_index_cache(cached)

    @staticmethod
    def _signature(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _read_index_cache(self) -> dict[str, dict[str, int | str]]:
        if self.index_cache is None or not self.index_cache.exists():
            return {}
        try:
            value = json.loads(self.index_cache.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index_cache(self, value: dict[str, dict[str, int | str]]) -> None:
        if self.index_cache is None:
            return
        self.index_cache.parent.mkdir(parents=True, exist_ok=True)
        self.index_cache.write_text(json.dumps(value), encoding="utf-8")

    def __len__(self) -> int:
        return sum(self.sizes)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        paths = self.paths[worker_id::worker_count]
        rng = np.random.default_rng(torch.initial_seed() + worker_id)
        if self.training:
            paths = list(paths)
            rng.shuffle(paths)
        for path in paths:
            with np.load(path) as data:
                positions = data["positions"].copy()
                starts = data["start_indices"].copy()
                ends = data["end_indices"].copy()
            indices = np.arange(len(positions))
            if self.training:
                rng.shuffle(indices)
            for index in indices:
                yield (
                    torch.from_numpy(positions[index]),
                    torch.tensor(starts[index], dtype=torch.long),
                    torch.tensor(ends[index], dtype=torch.long),
                )

    def set_training(self, training: bool) -> None:
        self.training = training


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, limit: int | None) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    start_top1 = start_top5 = end_top1 = end_top5 = 0
    complete_top1 = complete_top5 = complete_top10 = 0
    with torch.no_grad():
        for batch_index, (positions, starts, ends) in enumerate(loader):
            positions = positions.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            ends = ends.to(device, non_blocking=True)
            start_logits, end_logits = model(positions)
            loss = criterion(start_logits, starts) + criterion(end_logits, ends)
            total_loss += loss.item() * len(positions)
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
    return {
        "loss": total_loss / denominator,
        "start_top1": start_top1 / denominator,
        "start_top5": start_top5 / denominator,
        "end_top1": end_top1 / denominator,
        "end_top5": end_top5 / denominator,
        "complete_top1": complete_top1 / denominator,
        "complete_top5": complete_top5 / denominator,
        "complete_top10": complete_top10 / denominator,
        "complete_masked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Xiangqi Tiny-ResNet policy baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume from a saved checkpoint")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths = sorted(args.data_dir.glob("train-*.npz"))
    validation_paths = sorted(args.data_dir.glob("validation-*.npz"))
    test_paths = sorted(args.data_dir.glob("test-*.npz"))
    if not train_paths or not validation_paths or not test_paths:
        raise FileNotFoundError(f"train/validation/test NPZ shards not found under {args.data_dir}")

    index_cache = args.data_dir / ".shard_index.json"
    train_dataset = ShardDataset(train_paths, index_cache)
    validation_dataset = ShardDataset(validation_paths, index_cache)
    train_dataset.set_training(True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = TinyResNet(channels=64, blocks=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = max(1, (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * updates_per_epoch
    )
    criterion = nn.CrossEntropyLoss()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", checkpoint["validation_loss"]))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(json.dumps({
            "resumed_from": str(args.resume),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
        }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, ensure_ascii=False))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        epoch_batches = len(train_loader)
        for batch_index, (positions, starts, ends) in enumerate(train_loader):
            positions = positions.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            ends = ends.to(device, non_blocking=True)
            if args.mirror:
                positions = torch.flip(positions, dims=[3])
                starts = (starts // positions.shape[3]) * positions.shape[3] + (positions.shape[3] - 1 - starts % positions.shape[3])
                ends = (ends // positions.shape[3]) * positions.shape[3] + (positions.shape[3] - 1 - ends % positions.shape[3])
            start_logits, end_logits = model(positions)
            loss = (criterion(start_logits, starts) + criterion(end_logits, ends)) / args.accumulation_steps
            loss.backward()
            should_update = (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            completed_batches = batch_index + 1
            if completed_batches == 1 or completed_batches % 100 == 0 or completed_batches == epoch_batches:
                elapsed = time.perf_counter() - epoch_started
                batches_per_second = completed_batches / elapsed if elapsed else 0.0
                remaining = epoch_batches - completed_batches
                eta_seconds = remaining / batches_per_second if batches_per_second else 0.0
                memory = ""
                if device.type == "cuda":
                    memory = f" gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}GiB"
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
        validation_metrics = evaluate(model, validation_loader, device, limit=50 if args.max_steps else None)
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
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": vars(args),
        }
        torch.save(checkpoint, args.checkpoint_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.checkpoint_dir / "best.pt")
        print(json.dumps({"epoch": epoch + 1, "validation": validation_metrics, "step": global_step}, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement", flush=True)
            break
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    test_metrics = evaluate(model, DataLoader(ShardDataset(test_paths, index_cache), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0), device, limit=50 if args.max_steps else None)
    print(json.dumps({"test": test_metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
