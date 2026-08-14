# Xiangqi training project

This project follows `plan.md` for a 4 GB GTX 1650 Ti:

- PyTorch 2.x on Python 3.11
- Tiny-ResNet with four residual blocks and 64 channels
- Input shape `(14, 10, 9)`
- Separate 90-class start and end move heads
- First training pass uses FP32

## Setup

Install `uv` and synchronize the project environment in PowerShell:

```powershell
python -m pip install uv
uv sync
```

`uv sync` creates or updates `.venv` from `pyproject.toml` and `uv.lock`. The NVIDIA driver supplies compatibility for the packaged CUDA runtime; a separate CUDA Toolkit is not required for this project.

## Verify CUDA

```powershell
uv run python scripts\check_cuda.py
```

The check must report `CUDA available: True`, the GTX 1650 Ti, approximately 4 GiB of VRAM, and `CUDA verification passed.`
