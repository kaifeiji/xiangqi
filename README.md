# 象棋模型训练项目

本项目面向消费级显卡训练中国象棋模型，包含人类棋谱数据处理、ResNet/Pikafish 蒸馏训练、ONNX 导出、Rust 棋规引擎、MCTS 推理和 Web 对弈界面。

本文只保留复现实验所需入口。实现背景、调参过程和研究记录按主题放在 `docs/`。

## 复现环境

需要：

- Python 3.11/3.12
- Node.js/npm
- Rust stable GNU target（Windows 下使用 MSYS2 MINGW64 GCC）
- PyTorch 2.x；CUDA 训练需要匹配的 NVIDIA 驱动和 PyTorch CUDA 环境

安装 Python 与前端依赖：

```powershell
python -m pip install uv
python -m uv sync
npm ci
```

Windows Rust 工具链首次安装：

```powershell
C:\msys64\usr\bin\bash.exe -lc "pacman --noconfirm -S --needed mingw-w64-x86_64-toolchain"
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
```

Pikafish 只在引擎对手、标注和 CPU bench 中需要。复制 `.env.example` 为 `.env.local`，或设置：

```powershell
$env:PIKAFISH_PATH = "C:\path\to\pikafish.exe"
$env:PIKAFISH_NNUE_PATH = "C:\path\to\pikafish.nnue"
```

更多环境、ONNX Runtime、Web 服务和模型加载细节见 [docs/setup.md](docs/setup.md)。

## 复现实验

所有命令从仓库根目录执行。

### 1. 生成统一棋局 JSONL

将原始 PGN/XQF 数据放入 `data/raw/` 后运行：

```powershell
uv run python scripts\unify_format.py `
  --pgn data\raw\dpxq-99813games.pgns data\raw\WXF-41743games.pgns `
  --book-input-dir data\raw\xqp `
  --output data\processed\human_games `
  --split-ratio 98:1:1 `
  --shard-size 8192
```

续跑使用相同输入追加 `--resume`。数据格式、去重和 current-view 说明见 [docs/data-training.md](docs/data-training.md)。

### 2. 训练普通 ResNet 策略/value 模型

生成 NPY：

```powershell
uv run python scripts\prepare_data.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\dataset `
  --shard-size 8192
```

训练策略模型：

```powershell
uv run python scripts\train.py `
  --data-dir data\processed\dataset `
  --checkpoint-dir checkpoints\policy `
  --channels 128 --blocks 6 `
  --batch-size 256 --accumulation-steps 1 `
  --learning-rate 0.001 --epochs 100 --patience 5 `
  --num-workers 8 --mirror
```

带 value head 的训练在上述命令上添加 `--value-head`，并按显存调整 batch/accumulation。常用参数见 [docs/data-training.md](docs/data-training.md)。

### 3. 复现 Pikafish 蒸馏短实验

先做 Pikafish 标注 smoke：

```powershell
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\train-000.jsonl --output-dir data\processed\pikafish-smoke-train-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\validation-000.jsonl --output-dir data\processed\pikafish-smoke-validation-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\test-000.jsonl --output-dir data\processed\pikafish-smoke-test-1 --depth 10 --multipv 5
```

生成蒸馏 NPY：

```powershell
uv run python scripts\prepare_pikafish.py `
  --input-jsonl data\processed\pikafish-d10-m5 `
  --output-dir data\processed\pikafish-distillation `
  --max-candidates 5
```

当前可复现的 Pikafish 蒸馏基线是不带镜像的 20 epoch 训练：

```powershell
uv run python scripts\train_pikafish.py `
  --data-dir data\processed\pikafish-distillation\dataset `
  --checkpoint-dir checkpoints\pikafish-c192-b12-lr2e4-vlr2e5-vw1-vs450-w220-nomirror `
  --epochs 20 `
  --learning-rate 2e-4 --value-learning-rate 2e-5 --min-learning-rate 5e-6 `
  --warmup-steps 220 --weight-decay 1e-4 `
  --temperature 100 --value-scale 450 --policy-weight 1 --value-weight 1 `
  --micro-batch-size 2048 --global-batch-size 2048 `
  --max-grad-norm 1 --block-size 65536 `
  --num-workers 8 --prefetch-factor 4 --seed 42
```

该配置 20 epoch 已跑完，`best.pt` 对应 epoch 18；指标、训练趋势、旧 50-step smoke 和失效实验记录见 [docs/pikafish.md](docs/pikafish.md)。

### 4. 导出 ONNX 并启动 Web 对弈

```powershell
uv run python scripts\export_onnx.py checkpoints\policy\best.pt models\policy.onnx
cargo build --release --bin xiangqi-server
.\target\release\xiangqi-server.exe
```

另开终端：

```powershell
npm run dev
```

访问 `http://127.0.0.1:5173`。MCTS 和 Web 运行参数见 [docs/setup.md](docs/setup.md)，MCTS 实验记录见 [docs/mcts.md](docs/mcts.md)。

## 验证

修改代码后优先运行对应测试，再运行完整检查：

```powershell
uv run pytest -q
cargo test
npm run build
```

## 目录

- `data/`：原始数据、统一 JSONL、NPY 数据集和处理摘要。
- `checkpoints/`：训练输出和恢复点。
- `models/`：Web 服务加载的 ONNX 模型。
- `scripts/`：数据处理、训练、导出、检查脚本。
- `src/backend/`：Rust API、棋规引擎、MCTS 和 ONNX 推理。
- `src/frontend/`：React/Vite 前端源码。
- `src/frontend/public/`：Vite 静态资源目录，通过 `vite.config.ts` 的 `publicDir` 配置。

## 文档索引

- [docs/setup.md](docs/setup.md)：环境、Web 服务、模型加载、MCTS 环境变量和棋规限制。
- [docs/data-training.md](docs/data-training.md)：人类棋谱数据、统一格式、NPY 导出、普通训练和镜像增强。
- [docs/pikafish.md](docs/pikafish.md)：Pikafish benchmark、标注、蒸馏训练和 cp/mate 研究记录。
- [docs/mcts.md](docs/mcts.md)：MCTS 优化记录和当前搜索策略。
- [docs/architecture.md](docs/architecture.md)：Rust/Web 架构、前端存档格式、开局库和已废弃方向。
- [docs/experiments.md](docs/experiments.md)：训练、MCTS、benchmark 和已废弃方向的实验结论台账。
- [docs/references.md](docs/references.md)：文献和外部实现参考。
- [scripts/README.md](scripts/README.md)：脚本入口清单。
- [data/README.md](data/README.md)、[models/README.md](models/README.md)、[checkpoints/README.md](checkpoints/README.md)：目录约定。
