# 象棋模型训练项目

本项目面向消费级显卡训练中国象棋模型，使用人类棋局数据进行训练。

项目包含数据准备、ResNet 模型训练、Pikafish 蒸馏标签生成、模型评估和棋规引擎。数据准备阶段将不同来源的人类棋局统一转换为局面、走法标签和可选的终局价值标签；训练阶段使用统一的棋盘张量训练策略模型，也可以训练 value head。Web 服务支持模型推理、MCTS 和 Pikafish 对手。

- Python 3.11/3.12
- Node.js/npm（Web 前端开发和启动所需）
- PyTorch 2.x
- 输入张量：`(15, 10, 9)`
- ResNet 模型：起点/终点策略 logits，可选 value head

## 环境与依赖

```powershell
python -m pip install uv
python -m uv sync
```

### Rust 棋规引擎

Rust native 引擎使用 MSYS2 MINGW64 GCC，不依赖 Visual Studio。首次安装工具链：

```powershell
C:\msys64\usr\bin\bash.exe -lc "pacman --noconfirm -S --needed mingw-w64-x86_64-toolchain"
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
```

编译 Rust 后端：

```powershell
cargo build --release --bin xiangqi-server
```

仓库中的 `.cargo/config.toml` 已固定 GNU target 和 MINGW64 linker。验证：

```powershell
cargo test
```

当前 GNU Windows target 不支持 ort-sys 自动下载 runtime。首次运行请将 DLL 准备到项目根目录：

```powershell
.\target\release\xiangqi-server.exe
```

前端使用 Node.js/npm，开发模式启动前需要先安装依赖：

```powershell
npm ci
```

训练或本地推理使用 CUDA 时，需要兼容的 NVIDIA 驱动和 PyTorch CUDA 环境；CPU 模式不需要 CUDA。Pikafish 仅在引擎对手或标注流程中需要，不是 PyTorch 依赖。

本地 Pikafish 路径模板见 [`.env.example`](.env.example)。Rust Web 服务启动时会读取项目根目录的 `.env.local`；也可以显式设置进程环境变量：

```powershell
$env:PIKAFISH_PATH = "C:\path\to\pikafish.exe"
$env:PIKAFISH_NNUE_PATH = "C:\path\to\pikafish.nnue"
cargo build --release --bin xiangqi-server
.\target\release\xiangqi-server.exe
```

MCTS 默认批大小为 32，可在 `.env.local` 中通过 `MCTS_BATCH_SIZE` 调整；批大小越大，通常推理调用次数越少，但单批显存和延迟会增加：

```text
MCTS_BATCH_SIZE=32
```

MCTS 默认使用策略温度 `MCTS_POLICY_TEMPERATURE=1.25`，用于避免先验概率过度集中；设为 `1.0` 可恢复原始策略分布。

## Web 服务

项目提供 Rust Web 对弈服务。

启动服务：

```powershell
cargo build --release --bin xiangqi-server
.\target\release\xiangqi-server.exe
```

另开一个终端启动 Vite 开发服务器；它会将 API 代理到 `http://127.0.0.1:8000`，访问 `http://127.0.0.1:5173`：

```powershell
npm ci
npm run dev
```

后端端口可通过 `XIANGQI_PORT` 修改；修改后需要同步调整 Vite 代理配置。

Web 界面提供人机对弈、模型对弈和 JSON 存档回放。模型对弈支持自动/单步推进、模型选择和可选的 MCTS 时间预算；落子默认使用确定性策略（不启用温度采样）；人机模式支持悔棋和局面导航。

开局 book（内置主流开局）默认启用。模型对弈和人机对弈中的模型方会使用开局库；人机对弈不会替人类预先走第一着：

- 人类执黑时，模型执红，会从内置主流开局首着池随机选择一个局面作为起点。
- 人类执红时，从完整初始局面开始，等待人类走第一着。
- `model-model` 未提供 `fen` 时，仍从内置主流开局首着池随机选择一个局面作为起点。
- 当请求提供 `fen` 时，始终以该 `fen` 为准。

## 数据与训练

数据处理和训练流程为：`data/raw/` -> 统一 JSONL -> NPY 数据集 -> `checkpoints/`。支持 PGN、XQF 等输入格式；脚本参数、数据格式和训练配置统一见 [scripts/README.md](scripts/README.md)，目录约定见 [data/README.md](data/README.md)。

## 开发入口

前端源码位于 `src/frontend/`，使用 React、TypeScript 和 Vite；后端源码位于 `src/backend/`，由 Rust API、棋规引擎和 ONNX MCTS 组成。

单独检查前端构建：

```powershell
npm ci
npm run build
```

## 模型加载

训练完成后，将模型 checkpoint 放入 `models/` 目录或其子目录。服务会递归发现 `.pt`、`.pth` 和 `.ckpt` 文件，并在 Web 界面中使用相对路径作为模型 ID。训练脚本保存的 checkpoint 包含模型结构配置时，服务可以据此重建模型；不同的模型架构或 value head 应在文件名或目录名中明确区分。

Pikafish 不作为 PyTorch checkpoint 放入 `models/`，而是通过 `.env.local` 中的 `PIKAFISH_PATH` 和可选的 `PIKAFISH_NNUE_PATH` 配置。

## 棋规与终局限制

服务会处理将帅不存在、无合法走法、重复局面、长将、长捉、理论和棋以及自然限着。对局还设置 120 ply 的无吃子/无过河兵自然限着和 600 ply 的最大局长；具体结果会显示在对局状态中。

## 测试

运行全项目测试：

```powershell
uv run pytest -q
```

修改代码后，优先运行对应测试，再运行完整测试集。

数据准备、训练、评估和测试命令见 [scripts/README.md](scripts/README.md)。

其他目录说明：[`models/README.md`](models/README.md)、[`checkpoints/README.md`](checkpoints/README.md)。

## 文献参考

- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404)（Silver et al., Science, 2018）：介绍基于神经网络策略/价值双头与蒙特卡洛树搜索的自博弈系统。
- [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439)（Anthony, Tian, Barber, NeurIPS, 2017）：介绍 Expert Iteration，以及深度学习模型与树搜索之间的迭代训练流程。
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)（Hinton, Vinyals, Dean, 2015）：介绍使用温度参数生成软目标的知识蒸馏方法。
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)（Haarnoja et al., ICML, 2018）：介绍最大熵强化学习和基于 Boltzmann 分布的随机策略。
- [Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://doi.org/10.1145/3394486.3409195)（McIlroy-Young et al., KDD, 2020）：研究棋类人工智能与人类棋步行为的对齐和建模。
- [Not All Samples Are Created Equal: Deep Learning with Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html)（Katharopoulos, Fleuret, ICML, 2018）：研究基于样本信息量的深度学习重要性采样方法。
- [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)（Zinkevich et al., NeurIPS, 2007）：提出反事实后悔最小化方法，建立不完全信息博弈中的迭代求解框架。
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)（Hu, Shen, Sun, CVPR, 2018）：提出 Squeeze-and-Excitation 模块，通过通道间信息建模和通道重标定增强卷积网络表示能力。
- [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29)（Kocsis, Szepesvári, ECML, 2006）：提出 UCT，将置信上界策略用于蒙特卡洛树搜索中的节点选择。
- [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/Coulom-2006.pdf)（Coulom, 2006）：研究蒙特卡洛树搜索中的选择、模拟和回传机制。
- [Policy Distillation](https://arxiv.org/abs/1511.06295)（Rusu et al., 2015）：说明以软化动作目标蒸馏策略。
- [Pikafish types.h](https://github.com/official-pikafish/Pikafish/blob/master/src/types.h)：mate 与普通搜索值的内部编码。
- [Pikafish score.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/score.cpp)：普通值和 mate 在 UCI 输出边界的类型转换。
- [Pikafish uci.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/uci.cpp)：`score cp`、`score mate` 和 WDL 的格式化逻辑。
- [Pikafish engine.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/engine.cpp)：引擎选项，包括 `UCI_ShowWDL`。
- [Stockfish types.h](https://github.com/official-stockfish/Stockfish/blob/master/src/types.h)、[score.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/score.cpp) 与 [uci.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp)：国际象棋引擎的对应实现；其 tablebase 特殊编码不可直接迁用到象棋训练标签。
- [PyTorch Data Loading](https://docs.pytorch.org/docs/stable/data.html) 与 [DataLoader source](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)：map-style dataset、worker 与采样器行为。
- [UCI Protocol](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html)：UCI 协议说明与规范下载页。
