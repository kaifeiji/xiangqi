# 脚本入口

所有命令从仓库根目录执行。本文只列脚本职责；数据格式、训练参数和实验记录见 `docs/`。

## 数据处理

- `unify_format.py`：解析 PGN/XQF，回放校验棋局，输出按 split/shard 切分的统一 JSONL。详见 [docs/data-training.md](../docs/data-training.md)。
- `prepare_data.py`：从统一 JSONL 导出普通 ResNet 训练 NPY，包括 positions、start/end indices 和终局 value。
- `prepare_current_view.py`：导出 current-side-view 数据，使当前行棋方始终处于统一视角。
- `prepare_pikafish.py`：从 Pikafish canonical 标注导出 joint policy/value 蒸馏 NPY。详见 [docs/pikafish.md](../docs/pikafish.md)。

## 训练

- `train.py`：普通 ResNet policy/value 训练入口，支持 resume、mirror、current-view、AMP 和 smoke `--max-steps`。
- `train_pikafish.py`：Pikafish joint policy/value 蒸馏训练入口，支持 ragged 合法着、独立 value learning rate、自动 resume 和 progress JSONL。
- `training_models.py`：训练侧模型结构定义。

## Pikafish

- `benchmark_pikafish.py`：调用 Pikafish 内置 `bench`，比较本机不同 CPU 二进制的兼容性和 NPS。
- `annotate_pikafish.py`：使用 Pikafish MultiPV 为统一 JSONL 标注 teacher candidates、score 和 PV。

## 导出与检查

- `export_onnx.py`：将训练 checkpoint 导出为 Web 服务可加载的 ONNX。
- `check_cuda.py`：检查 PyTorch CUDA 可见性。
- `data_utils.py`：脚本共享的数据解析、环境变量和坐标转换工具。

## 常用帮助

```powershell
uv run python scripts\unify_format.py --help
uv run python scripts\prepare_data.py --help
uv run python scripts\train.py --help
uv run python scripts\annotate_pikafish.py --help
uv run python scripts\prepare_pikafish.py --help
uv run python scripts\train_pikafish.py --help
uv run python scripts\export_onnx.py --help
uv run python scripts\benchmark_pikafish.py --help
```

复现实验的最短路径见 [README.md](../README.md)。分类细节见 [docs/data-training.md](../docs/data-training.md)、[docs/pikafish.md](../docs/pikafish.md)、[docs/setup.md](../docs/setup.md) 和 [docs/mcts.md](../docs/mcts.md)。
