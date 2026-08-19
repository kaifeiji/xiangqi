# 脚本、数据与训练

所有命令从仓库根目录执行。项目环境使用 `uv` 管理：

```powershell
python -m pip install uv
python -m uv sync
```

Pikafish 路径参考仓库根目录的 [`.env.example`](../.env.example)。复制为 `.env.local` 后，脚本会自动读取其中的路径变量。

## CUDA 检查

```powershell
uv run python scripts\check_cuda.py
```

## 人类棋局数据

输入数据统一经过结构化解析、走法规范化和完整棋局回放。损坏记录不会进入训练数据；来源、棋局编号、结果和走法元数据保留在统一 JSONL 中。

统一编码约定：棋盘张量为 `(15, 10, 9)` 的 `float32`，走法起点和终点索引范围为 `0~89`。数据按棋局切分，避免同一棋局同时出现在 train、validation 和 test。

### 统一中间格式

`unify_format.py` 是唯一的原始数据解析入口。目前处理 PGN 和 XQF，并输出按 split 和 shard 保存的逐局 JSONL：

```text
PGN/XQF
  -> train-000.jsonl
  -> validation-000.jsonl
  -> test-000.jsonl
```

每条记录包含初始 FEN、规范化 ICCS 走法、结果、来源信息、去重 ID 和审计元数据。重复棋局写入 `duplicates.jsonl`，不会静默丢弃。

```powershell
uv run python scripts\unify_format.py `
  --pgn data\raw\dpxq-99813games.pgns data\raw\WXF-41743games.pgns `
  --book-input-dir data\raw\chess_book-main `
  --output data\processed\human_games `
  --shard-size 8192
```

续跑时使用相同输入和 `--resume`：

```powershell
uv run python scripts\unify_format.py `
  --pgn data\raw\dpxq-99813games.pgns data\raw\WXF-41743games.pgns `
  --book-input-dir data\raw\chess_book-main `
  --output data\processed\human_games `
  --shard-size 8192 `
  --resume
```

`--max-games 1 --shard-size 1` 可用于小规模 smoke。统一阶段不调用 Pikafish，也不展开逐 ply 样本。

### 去重与来源

去重键由规范化初始 FEN、完整走法序列和结果组成。不同来源的相同棋局只保留一份，重复记录写入 `duplicates.jsonl`。发布数据集或模型前，应单独核实各来源棋谱和解析工具的许可范围。

## 数据准备与模型训练

统一 JSONL 生成后，`prepare_data.py` 读取文件或 shard 目录，生成最终策略训练 NPY：

```powershell
uv run python scripts\prepare_data.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\dataset `
  --shard-size 8192
```

输出字段：

```text
train-000-positions.npy
train-000-start_indices.npy
train-000-end_indices.npy
train-000-values.npy
validation-000-*.npy
test-000-*.npy
```

普通策略模型训练：

```powershell
uv run python scripts\train.py `
  --data-dir data\processed\dataset `
  --checkpoint-dir checkpoints\policy `
  --channels 128 --blocks 6 `
  --batch-size 256 --accumulation-steps 1 `
  --learning-rate 0.001 --epochs 100 --patience 5 `
  --num-workers 8 --mirror
```

带 value head 的训练：

```powershell
uv run python scripts\train.py `
  --data-dir data\processed\dataset `
  --checkpoint-dir checkpoints\resnet-value `
  --channels 64 --blocks 4 `
  --batch-size 16 --accumulation-steps 8 `
  --epochs 100 --patience 5 --value-head
```

训练会在 validation 上评估并早停，结束后输出 test 指标。训练前应检查标签范围、棋盘形状、样本对齐和 split 是否按棋局隔离。训练完成后，checkpoint 可用于 Web 服务。

`train.py` 还支持以下常用选项：

- `--current-view`：将输入和走法统一到当前行棋方视角，适用于 `prepare_current_view.py` 生成的数据。
- `--resume CHECKPOINT`：从已有 checkpoint 恢复模型、优化器和训练进度。
- `--max-steps`：限制优化步数，适合 smoke 或短实验。
- `--amp`/`--no-amp`：启用或禁用 CUDA 混合精度；默认启用。
- `--prefetch-factor`：配置 DataLoader worker 的预取 batch 数。
- `--mirror`：启用棋盘镜像增强。

训练输出目录通常包含 `last.pt`、`best.pt` 和 `metrics.jsonl`。前两者分别表示最近一次保存和 validation 指标最佳的 checkpoint，指标文件用于记录训练/验证过程。

### Current-side-view 数据

如果希望让模型始终从当前行棋方视角读取棋盘，可使用：

```powershell
uv run python scripts\prepare_current_view.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\current_view `
  --shard-size 8192
```

输出目录包含 `dataset/` 下的 positions、起点、终点和 value NPY shard，以及 `dataset_summary.json` 和 `processed_game_ids.txt`。训练时同时指定 `--data-dir data\processed\current_view` 和 `--current-view`。该转换只读取统一 JSONL，可通过 `--max-games` 做 smoke。

## Pikafish 蒸馏

`annotate_pikafish.py` 只读取统一 JSONL，不直接解析 PGN 或 XQF。它对每个存在人类走法的局面执行 Pikafish MultiPV 搜索，并写入 canonical JSONL。标注阶段不生成 NPY；后续按所需训练配置使用数据准备脚本生成 NPY。

```powershell
uv run python scripts\annotate_pikafish.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\pikafish-d10-m5 `
  --depth 10 `
  --multipv 5 `
  --pikafish-threads 1
```

搜索预算可用 `--depth N` 或 `--movetime-ms N` 二选一。`--workers N` 用于并行处理输入 shard 目录；单个 JSONL 文件不支持多 worker。长任务可使用 `--resume`，失败游戏默认跳过，使用 `--retry-failed` 才会重试失败记录。

Pikafish 从 `.env.local` 读取：

```env
PIKAFISH_PATH=C:\workspace\Pikafish.2026-01-02\Windows\pikafish-avx2.exe
PIKAFISH_NNUE_PATH=C:\workspace\Pikafish.2026-01-02\pikafish.nnue
```

每条记录的 `schema_version` 为 `1`，包含 `game_id`、`split`、`ply`、当前 FEN、人类 ICCS 走法和 `teacher`。teacher 的 `score_kind`/`score` 是 PikaFish 原始的**当前行棋方**分数；`cp` 与带符号 mate 距离都不在标注阶段归一化。每个 MultiPV candidate 包含 rank、首步、原始分数、depth、nodes 和完整 ICCS PV。所有候选 PV 都会回放校验合法性，PV1 必须与 `bestmove` 一致。

训练期可从有效候选的相对 score 构造 soft policy：

$$
p_i = \operatorname{softmax}((s_i - s_{\max}) / \tau)
$$

MultiPV 分差可作为裁剪后的 policy 难度权重。PV2/PV3 不混入 PV1 value target；未进入 Top-N 的合法走法不应直接当作非法动作。cp/mate 到 value、mate 的 pseudo-cp 映射、softmax 温度与人类/teacher policy 混合权重均在训练准备或训练阶段配置。

目录中会写入 `annotation_config.json`、`dataset_summary.json`、按输入 shard 命名的 JSONL 和对应的失败事件流。每个 shard 先写入 `.partial.jsonl`，完成后原子改名；每局写入后 fsync。`--resume` 会扫描 JSONL，只接受 ply 从 0 连续且数量与源棋局一致的完整游戏，并移除 partial 中断尾。普通续跑跳过已失败游戏；使用 `--retry-failed` 才会重试尚未成功的失败游戏。

输出目录绑定 schema、输入 shard 的 SHA-256、depth/时间预算、MultiPV 和 PikaFish 线程数。配置不匹配时会拒绝运行，需使用新的 `--output-dir`。

先分别对各 split 跑 1 局 smoke：

```powershell
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\train-000.jsonl --output-dir data\processed\pikafish-smoke-train-1 --depth 10 --multipv 5 --max-games 1
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\validation-000.jsonl --output-dir data\processed\pikafish-smoke-validation-1 --depth 10 --multipv 5 --max-games 1
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\test-000.jsonl --output-dir data\processed\pikafish-smoke-test-1 --depth 10 --multipv 5 --max-games 1
```

确认三次运行零失败、PV 回放和恢复语义正确后，再将每个命令的 `--max-games` 改为 `10`，并添加 `--resume`。
