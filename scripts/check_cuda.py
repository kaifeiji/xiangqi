from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in the active Python environment.")
        return 2

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda or 'none'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("CUDA verification failed: PyTorch cannot access an NVIDIA GPU.")
        return 1

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"device: {properties.name}")
    print(f"compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")

    left = torch.randn((16, 14, 10, 9), device=device)
    right = torch.randn((16, 14, 10, 9), device=device)
    result = left @ right.transpose(-1, -2)
    torch.cuda.synchronize()
    print(f"GPU tensor check: {tuple(result.shape)}")
    print("CUDA verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
