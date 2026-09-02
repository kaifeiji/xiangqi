# 数据与普通训练

本文整理人类棋谱数据、统一格式、NPY 导出和普通 ResNet 训练。Pikafish 标注与蒸馏见 [pikafish.md](pikafish.md)。

所有命令从仓库根目录执行。

## 原始数据

输入数据统一经过结构化解析、走法规范化和完整棋局回放。损坏记录不会进入训练数据；来源、棋局编号、结果和走法元数据保留在统一 JSONL 中。

原始棋谱来源：

- [CGLemon/chinese-chess-PGN](https://github.com/CGLemon/chinese-chess-PGN)：中国象棋 PGN 棋谱。
- [weiyinfu/xqp](https://github.com/weiyinfu/xqp)：中国象棋棋谱及相关数据。

使用这些数据前，请分别核实原仓库的授权条款和数据使用范围，并将下载内容放入 `data/raw/`。

统一编码约定：棋盘张量为 `(15, 10, 9)` 的 `float32`，走法起点和终点索引范围为 `0..89`。数据按棋局切分，避免同一棋局同时出现在 train、validation 和 test。

## 统一中间格式

`unify_format.py` 是原始数据解析入口。目前处理 PGN 和 XQF，并输出按 split 和 shard 保存的逐局 JSONL：

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
  --book-input-dir data\raw\xqp `
  --output data\processed\human_games `
  --split-ratio 98:1:1 `
  --shard-size 8192
```

续跑时使用相同输入和 `--resume`。默认按棋局 ID 哈希稳定划分 train/validation/test，可用 `--split-ratio TRAIN:VALIDATION:TEST` 自定义比例。`--max-games 1 --shard-size 1` 可用于 smoke。

## Smoke 与容量规划

首次接入新数据源时，不要直接全量跑。推荐用小参数验证三件事：解析能完成、回放合法、split/shard 输出正确。

```powershell
uv run python scripts\unify_format.py `
  --pgn data\raw\sample.pgn `
  --output data\processed\human_games_smoke `
  --max-games 1 `
  --shard-size 1
```

容量和耗时估算应按“局面数”而不是“棋局数”。棋局长度差异很大，长局会带来更多训练样本和更高标注成本。统一 JSONL、NPY 和 Pikafish 标注的每局面大小不同，因此每次更改 schema 后应先用 smoke 输出目录实测：

```powershell
Get-ChildItem data\processed\human_games_smoke -Recurse | Measure-Object Length -Sum
```

用 smoke 的 `bytes/position` 乘以预计 positions，比按文件数量估算可靠。

## 终局 value 数据集

统一 JSONL 生成后，`prepare_data.py` 读取文件或 shard 目录，生成普通策略/value 训练 NPY：

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

`train.py` 常用选项：

- `--current-view`：将输入和走法统一到当前行棋方视角。
- `--resume CHECKPOINT`：恢复模型、优化器和训练进度。
- `--max-steps`：限制优化步数，适合 smoke 或短实验。
- `--amp`/`--no-amp`：启用或禁用 CUDA 混合精度；默认启用。
- `--prefetch-factor`：配置 DataLoader worker 预取 batch 数。
- `--mirror`：启用棋盘镜像增强。

训练输出目录通常包含 `last.pt`、`best.pt` 和 `metrics.jsonl`。

## Current-side-view 数据

如果希望模型始终从当前行棋方视角读取棋盘，可使用：

```powershell
uv run python scripts\prepare_current_view.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\current_view `
  --shard-size 8192
```

输出目录包含 `dataset/` 下的 positions、起点、终点和 value NPY shard，以及 `dataset_summary.json` 和 `processed_game_ids.txt`。训练时同时指定 `--data-dir data\processed\current_view` 和 `--current-view`。

## Mirror 数据增强

`train.py --mirror` 在每个训练 batch 内追加左右镜像样本，不生成新的 NPY shard。它沿棋盘宽度翻转 positions，并把每个 start/end 格点从 `9r+c` 映射为 `9r+(8-c)`；value 标签直接复制。validation、test、推理均不镜像。

镜像会使该 batch 的实际前向/反向样本数加倍，因此显存和训练时间也随之增加；`--batch-size` 应按未镜像样本理解。`train_pikafish.py --mirror` 会同步镜像 8100 联合动作 ID、候选动作与 ragged 合法着集合。

## 合法着 Mask 的分工

人类棋局里的监督标签本身应是合法着，所以普通 `train.py` 的训练 loss 不需要再对完整动作空间做合法着 hard mask。这样训练链路更简单，也避免把棋规生成成本放进每个训练 batch。

validation/test 的完整着法指标仍应保留合法着语义：`complete_top1/top5/top10` 用来衡量模型在完整合法着集合中的命中情况，比单独 start/end top-k 更接近真实落子质量。当前 `train.py` 的指标显式记录 `complete_masked=false`，表示普通训练/验证未启用完整合法着 mask。

Pikafish 蒸馏训练不同：它天然依赖 teacher candidate 与全合法着集合的归一化，`train_pikafish.py` 会读取 ragged `legal_action_ids`，只在合法着上计算 policy softmax。

## 检查点

`checkpoints/` 每个子目录对应一组训练实验。常见文件：

- `best.pt`：validation 指标最佳 checkpoint。
- `last.pt`：最近一次保存的 checkpoint，可用于中断后恢复。
- `metrics.jsonl`：`train.py` 的逐 epoch/step 训练、验证和测试指标记录。
- `progress.jsonl`：`train_pikafish.py` 的追加事件流。
- `epoch-xxxx.pt`：`train_pikafish.py` 每个完成 epoch 的编号 checkpoint。
