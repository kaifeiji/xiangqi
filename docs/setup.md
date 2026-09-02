# 环境与服务

本文保存项目运行环境、Web 服务和模型加载细节。复现实验的最短命令见根目录 README。

## Python 与前端

项目 Python 环境使用 `uv` 管理：

```powershell
python -m pip install uv
python -m uv sync
```

前端使用 Node.js、React、TypeScript 和 Vite：

```powershell
npm ci
npm run build
```

## Rust 棋规引擎

Windows 下 Rust native 后端使用 MSYS2 MINGW64 GCC，不依赖 Visual Studio。首次安装工具链：

```powershell
C:\msys64\usr\bin\bash.exe -lc "pacman --noconfirm -S --needed mingw-w64-x86_64-toolchain"
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
```

仓库中的 `.cargo/config.toml` 固定 GNU target 和 MINGW64 linker。验证：

```powershell
cargo test
```

GNU Windows target 不支持 `ort-sys` 自动下载 runtime。服务启动和 `/health` 不会强制加载 ONNX Runtime；首次使用 ONNX 模型推理时，把匹配 `ort` crate `api-22` feature 的 ONNX Runtime 1.22.x `onnxruntime.dll` 放在服务程序同级目录，或放入其 `lib/` 子目录。

```powershell
Copy-Item C:\path\to\onnxruntime.dll .\target\release\onnxruntime.dll
```

## Pikafish 环境变量

Pikafish 只在引擎对手、标注流程和 CPU bench 中需要。复制 `.env.example` 为 `.env.local`，或显式设置进程环境变量：

```powershell
$env:PIKAFISH_PATH = "C:\path\to\pikafish.exe"
$env:PIKAFISH_NNUE_PATH = "C:\path\to\pikafish.nnue"
```

Rust Web 服务启动时会读取项目根目录的 `.env.local`。

## MCTS 运行参数

以下变量可放入 `.env.local`：

```text
MCTS_BATCH_SIZE=8
MCTS_POLICY_TEMPERATURE=1.25
MCTS_EXPLORATION=1.25
MCTS_Q_GUARD_MIN_VISITS=25
MCTS_Q_GUARD_MIN_GAP=0.15
```

`MCTS_BATCH_SIZE` 控制批量叶节点评估大小。`MCTS_POLICY_TEMPERATURE` 用于避免策略先验过度集中。`MCTS_EXPLORATION` 是 PUCT 探索常数。`MCTS_Q_GUARD_MIN_VISITS` 与 `MCTS_Q_GUARD_MIN_GAP` 控制根节点最终选着的 Q 兜底：当 Q 最佳候选访问数达到下限，且当前方 Q 比 visits 第一高出阈值时，最终选 Q 最佳候选。

## Web 服务

启动后端：

```powershell
cargo build --release --bin xiangqi-server
.\target\release\xiangqi-server.exe
```

另开终端启动 Vite 开发服务器；它会把 API 代理到 `http://127.0.0.1:8000`：

```powershell
npm run dev
```

访问 `http://127.0.0.1:5173`。后端端口可通过 `XIANGQI_PORT` 修改；修改后需同步调整 Vite 代理配置。

Web 界面提供人机对弈、模型对弈、模型基准和 JSON 存档回放。模型方默认使用确定性策略；MCTS simulation 数可在界面中选择。

Vite 静态资源放在 `src/frontend/public/`，并通过 `vite.config.ts` 的 `publicDir` 指向该目录。放在这里的文件仍以根路径访问，例如 `/favicon.svg`。

## 模型加载

训练生成的 `.pt` checkpoint 不能直接被 Rust Web 服务加载，需要先导出 ONNX：

```powershell
uv run python scripts\export_onnx.py checkpoints\pikafish-c192-b12\best.pt models\pikafish-c192-b12.onnx
```

导出的 `.onnx` 放在 `models/` 或其子目录。服务递归扫描 `.onnx` 文件，并用相对 `models/` 的路径作为模型 ID。Pikafish NNUE 不放在 `models/`，只通过 `PIKAFISH_PATH` 和 `PIKAFISH_NNUE_PATH` 配置。

## 棋规与终局限制

服务处理将帅不存在、无合法走法、理论和棋、三次重复、120 ply 无吃子自然限着和 600 ply 最大局长。

当前不实现复杂长将/长捉归责。为避免模型陷入循环，模型方选着使用简化反循环规则：如果存在不会回到历史局面的合法着法，就只在这些非重复着法中选择；只有全部合法着法都会重复时，才允许重复。三次重复仍作为和棋终局兜底。
