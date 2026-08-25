from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from backend.models import PikafishResNet


def measure_batch(
    *,
    batch_size: int,
    channels: int,
    blocks: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float | int | bool]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = PikafishResNet(channels=channels, blocks=blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    total_steps = warmup_steps + measure_steps
    started = 0.0

    for step in range(total_steps):
        positions = torch.randn(batch_size, 15, 10, 9, device=device)
        actions = torch.randint(8100, (batch_size,), device=device)
        values = torch.empty(batch_size, device=device).uniform_(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            policy_logits, predictions = model(positions)
            loss = nn.functional.cross_entropy(policy_logits, actions)
            loss = loss + nn.functional.smooth_l1_loss(predictions.reshape(-1), values, beta=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == warmup_steps - 1:
            torch.cuda.synchronize(device)
            started = time.perf_counter()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    total_memory = torch.cuda.get_device_properties(device).total_memory
    peak_reserved = torch.cuda.max_memory_reserved(device)
    result: dict[str, float | int | bool] = {
        "batch_size": batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "positions_per_second": batch_size * measure_steps / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "total_memory_gib": total_memory / 1024**3,
        "has_ten_percent_headroom": peak_reserved <= total_memory * 0.9,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe CUDA training capacity for the Pikafish joint-policy ResNet.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the micro-batch probe")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes must be positive")
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("--warmup-steps and --measure-steps must be positive")

    results: list[dict[str, float | int | bool]] = []
    for batch_size in sorted(set(args.batch_sizes)):
        try:
            result = measure_batch(
                batch_size=batch_size,
                channels=args.channels,
                blocks=args.blocks,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "out_of_memory": True}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    accepted = [
        result for result in results
        if not result.get("out_of_memory") and result["has_ten_percent_headroom"]
    ]
    recommendation = max(accepted, key=lambda result: int(result["batch_size"])) if accepted else None
    print(json.dumps({"recommended_micro_batch": recommendation}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())