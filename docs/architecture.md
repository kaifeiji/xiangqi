# 架构与产品约定

本文记录从历史会话中沉淀出的长期约定。复现实验命令见根目录 README；MCTS 性能记录见 [mcts.md](mcts.md)。

## 当前架构

```text
React/Vite frontend
  -> Rust Axum API
  -> Rust 棋规、规则状态、MCTS
  -> ONNX Runtime / Pikafish UCI process
```

后端已转向 Rust native。历史上尝试过 Python 调用 native 棋规核心，但逐节点 FFI 会把 Python `Position` 转 FEN、跨边界后再解析回 Rust，MCTS 中会放大该成本。当前边界改为：Web/API 提交局面、模型和预算，搜索与规则状态整体留在 Rust 内部。

## Rust 工程边界

Rust 工程位于仓库根目录，使用 `.cargo/config.toml` 固定 Windows GNU target 与 MINGW64 linker。Windows 环境优先使用 MSYS2 MINGW64 GCC，不引入 Visual Studio 作为默认依赖。

Rust 侧负责：

- FEN 解析与棋盘增量状态。
- 合法着生成与将军检查。
- 终局与规则历史。
- ONNX 模型推理和 MCTS。
- Pikafish 作为 UCI 子进程的对手。

Python 侧保留为离线数据处理、训练和导出 ONNX，不参与 Web 对弈热路径。

## 前端静态资源

Vite 静态资源目录为 `src/frontend/public/`，由 `vite.config.ts` 的 `publicDir` 配置。该目录中的文件仍按根路径访问，例如 `/favicon.svg`。

需要由 TypeScript/React import 的资源应放入 `src/frontend` 下的源码目录；需要原样复制并以固定 URL 访问的资源放入 `src/frontend/public/`。

## Web 存档格式

前端保存的 JSON 存档使用蛇形字段 `initial_fen`，并把它放在 `snapshots` 上方：

```json
{
  "savedAt": "2026-09-02T00:00:00.000Z",
  "mode": "model-model",
  "humanSide": null,
  "initial_fen": "...",
  "snapshots": []
}
```

`snapshots` 每项保存：`fen`、`side_to_move`、`turn`、`rule60`、`result`，以及可选的 `mcts_debug` 或 `policy_debug`。

回放加载时若 `initial_fen` 的棋盘与第一个 snapshot 不同，会额外插入一个 `archive-initial` 初始局面，并停在 `position=0`，不会自动走第一步。这是为了兼容开局库：模型对弈的第一个 snapshot 可能已经包含红方开局首着。

## 行棋记录与分析面板

行棋记录从相邻 snapshot 的棋盘差分推导，而不是依赖后端额外保存 move 字段。模型分析面板使用 `mcts_debug`/`policy_debug`：

- `selected_move` 表示本次搜索最终落子。
- `root_children[].q` 表示对应根候选的搜索后平均 value。
- `root_network_value` 是模型对根局面的直接 value，不是某个具体 move 的价值。

Benchmark 列表不显示完整着法串，避免超长文本污染界面；详细走法保留在保存的 benchmark JSON 中。

## 开局库行为

模型方会在完整初始局面且轮到红方时使用内置主流开局首着池。人机对弈不会替人类预走第一着：

- 人类执红：从完整初始局面开始。
- 人类执黑：模型执红，会先从开局库走红方首着。
- 模型对弈未提供 FEN：默认可使用开局库首着。
- 请求提供 FEN：始终以请求 FEN 为准。

前端行棋记录必须把这类开局首着作为第 1 步显示，不能把开局后局面误当“初始局面”。

## Benchmark 约定

模型 benchmark 是后端后台任务，不依赖棋盘 UI 驱动。赛制固定为主流开局库成对换色：每个开局 A/B 各执红一次，因此总盘数为 `MAINSTREAM_OPENINGS.len() * 2`。这比“完全禁书”更稳定，因为纯策略模型常会集中选择相同首着，导致样本多样性不足。

Benchmark 结果按本地开始时间保存到 `benchmark/`，文件名形如 `20260902-123456-789.json`，可通过 `BENCHMARK_PATH` 改目录。JSON 中保留 `moves` 和 `final_fen` 供排障；前端卡片不显示完整着法串。

运行中 benchmark 前端每 5 秒刷新一次；无运行中任务时不轮询。列表按开始时间倒序，最新任务在上。

## 已废弃的方向

Alpha-beta + ResNet value 的运行时组合已撤掉。原因是传统 alpha-beta 搜索通常需要每秒极大量节点，当前 ResNet value 调用成本无法满足这种节点吞吐。相关离线准备能力可保留，但 Web 对弈主路径使用 policy/value + MCTS。

Python MCTS + Rust 棋规逐节点 FFI 也不作为方向继续推进。若引入 native 加速，应把搜索边界整体移入 Rust，而不是在 MCTS 内层反复跨语言调用。