# 象棋走法预测训练项目

本项目面向 4 GB NVIDIA GeForce GTX 1650 Ti 训练象棋走法预测模型。
- Python 3.11 和 PyTorch 2.x
- Tiny-ResNet：64 个通道、4 个残差块、约 134 万参数
- 输入张量形状：`(15, 10, 9)`，包含 14 个棋子类型通道和 1 个轮到行棋方通道
- 分别预测 90 类起点与 90 类终点
- 首轮训练使用 FP32

## 环境准备

在 PowerShell 中安装 `uv` 并同步项目环境：
```powershell
python -m pip install uv
python -m uv sync
```

`uv sync` 会依据 `pyproject.toml` 和 `uv.lock` 创建或更新 `.venv`。显卡驱动会为随 PyTorch 打包的 CUDA 运行时提供兼容性，不需要单独安装 CUDA Toolkit。

## 验证 CUDA

```powershell
python -m uv run python scripts\check_cuda.py
```

输出应包含 `CUDA available: True`、GTX 1650 Ti、约 4 GiB 显存以及 `CUDA verification passed.`。

## Web App 对弈

项目包含一个 Vite + TypeScript 可视化象棋客户端，支持两种模式：

- `human-model`
- `model-model`

使用一个命令构建 TypeScript 前端并启动后端：

```powershell
python -m uv run xiangqi-play --host 127.0.0.1 --port 8000
```

该命令会在项目根目录执行 `npm ci` 和 `npm run build`，随后由 Flask 在同一端口提供前端和 `/api`。浏览器打开 `http://127.0.0.1:8000`。

### 前端开发

开发前端时使用：

```powershell
python -m uv run xiangqi-play --dev
```

该命令会同时启动 Flask API（`http://127.0.0.1:8000`）和 Vite 开发服务器（`http://127.0.0.1:5173`）。开发时打开 `http://127.0.0.1:5173`；保存 `src/frontend/` 中的 React、TypeScript 或 CSS 文件后，Vite 会自动热更新页面。Vite 会将 `/api` 请求代理至 Flask，无需修改 API 地址。

HMR 使用 Vite 的 WebSocket，因此开发模式使用独立的 `5173` 端口；生产模式仍由 Flask 在单个 `8000` 端口提供页面和 API。

主要功能：

- 可视化棋盘和点选走子
- 人机模式可选择人类执红或执黑，并从 `models/` 选择一个模型
- 模型对弈可分别为红、黑双方从 `models/` 选择模型
- `model-model` 支持“模型走一步”
- 使用 `xiangqiground` 渲染棋盘和棋子，并显示最近一步走子

将训练生成的 `.pt`、`.pth` 或 `.ckpt` checkpoint 放入 `models/`（可使用子目录），刷新页面后即可在模型选择框中选取。

`src/backend/` 包含 Python 服务、推理和棋局逻辑；`src/frontend/` 包含 TypeScript 客户端源码。根目录包含 Node/Vite 配置。客户端使用 GPL-3.0-or-later 许可，因为其依赖 `xiangqiground`。

## 数据准备

默认原始棋谱路径：

- `data/raw/dpxq-99813games.pgns`
- `data/raw/WXF-41743games.pgns`

扫描棋谱文件和标签统计，输出到 `artifacts/scan/`：

```powershell
uv run python scripts\prepare_data.py --scan-only --output-dir artifacts\scan_xxx
```

`artifacts/scan/data_scan.json` 包含每个输入文件的 SHA-256、文件大小、棋局数、棋谱格式和结果分布，以及缺失 FEN、无效标签等统计信息。

执行结构校验，输出到 `artifacts/validate/`：

```powershell
uv run python scripts\prepare_data.py --validate --output-dir artifacts\validate_xxx
```

该命令生成：

- `validated_games.jsonl`：通过结构校验的棋局记录，包含标签、着法和源文件信息。
- `data_errors.jsonl`：未通过校验的棋局及错误原因。
- `validation_summary.json`：有效和错误棋局数量，以及校验范围说明。

导出训练数据集到 `data/processed/`：

```powershell
uv run python scripts\prepare_data.py --export --output-dir data\processed --shard-size 4096
```

导出目录内容：

- `data/processed/dataset_summary.json`：已处理棋局数、去重和跳过数量、各数据集样本数和分片数、输入文件哈希及 80/10/10 切分规则。
- `data/processed/dataset/train-*-positions.npy`、`train-*-start_indices.npy`、`train-*-end_indices.npy`：训练分片；验证和测试分片使用同一命名规则。三个数组均为未压缩 NPY，以便训练进程内存映射读取并直接产出完整 batch。全量数据约需 36 GiB 磁盘空间。
- 对应的 `*.jsonl`：每个样本的源文件、棋局、回合和结果等元数据。

可用 `--input` 指定一个或多个非默认棋谱文件；`--max-games` 可限制处理棋局数，用于冒烟测试。

将现有的压缩 NPZ 数据集转换为可内存映射的 NPY，而不重新解析棋谱：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --convert-npz data\processed\dataset
```

该命令原地新增 NPY 文件并保留 NPZ。若转换中断，删除不完整的 NPY 文件后重跑；只有在确认需要替换已有 NPY 时才添加 `--overwrite`。

## 训练

从导出的完整数据集开始训练，checkpoint 输出到 `checkpoints/run-1/`：

```powershell
uv run python scripts\train.py --data-dir data\processed\dataset --checkpoint-dir checkpoints\run-1 --batch-size 128 --accumulation-steps 1 --learning-rate 0.001 --epochs 100 --patience 5 --num-workers 8 --mirror
```

中断后从最近 checkpoint 续跑：

```powershell
uv run python scripts\train.py --data-dir data\processed\dataset --checkpoint-dir checkpoints\run-1 --batch-size 128 --accumulation-steps 1 --learning-rate 0.001 --epochs 100 --patience 5 --num-workers 8 --mirror --resume checkpoints\run-1\last.pt
```

训练输出：

- 控制台：设备、样本数、模型参数量；每 100 个 batch 的 loss、step、速度、剩余时间和 CUDA 显存；每轮验证指标和最终测试指标。
- `checkpoints/run-1/last.pt`：每轮结束时覆盖保存的最新模型、优化器、学习率调度器、epoch、step、验证指标和训练配置。
- `checkpoints/run-1/best.pt`：验证 loss 刷新最佳时保存的 checkpoint。
- `checkpoints/run-1/metrics.jsonl`：每轮追加一行 JSON，包含 epoch、step、平均训练 loss、验证指标、学习率和本轮耗时，可用于绘制训练曲线。
- 旧版 `.npz` 分片仍可读取，但会因解压产生额外 CPU 开销；重新导出数据后会自动优先使用新的 NPY 分片。

验证 loss 连续 `--patience` 个 epoch 未改善时会提前停止。当前训练和验证均未使用规则合法性掩码；完整走子 top-1、top-5 和 top-10 指标反映的是未掩码预测结果。

## 脚本说明

| 脚本 | 用途 | 输入与输出 |
| --- | --- | --- |
| `scripts/check_cuda.py` | 验证当前 Python 环境能否使用 CUDA，并执行一次 GPU 张量计算。 | 输出 Python、PyTorch、CUDA、显卡和显存信息；失败时返回非零状态码。 |
| `scripts/prepare_data.py` | 解析 PGNS 棋谱，执行扫描、结构校验或导出训练数据。 | 读取 `data/raw/` 下默认棋谱；导出可内存映射的 NPY 分片。 |
| `scripts/train.py` | 读取导出的分片，训练并评估 Tiny-ResNet。 | 优先读取 NPY 分片并以完整 batch 传递给训练循环；兼容旧 NPZ。 |
| `scripts/data_encoding.py` | 提供 FEN 编码、ICCS 坐标转换和棋局位置更新函数。 | 被数据准备、训练相关代码和自动化测试导入，不单独执行。 |

## 测试

编码测试覆盖 FEN 编码、ICCS 坐标往返转换、走子更新和轮到行棋方通道切换：

```powershell
uv run pytest
```
