# 脚本、数据与训练

所有命令从仓库根目录执行。项目环境使用 `uv` 管理：

```powershell
python -m pip install uv
python -m uv sync
```

Pikafish 路径参考仓库根目录的 [`.env.example`](../.env.example)。复制为 `.env.local` 后，`annotate_pikafish.py` 与 `benchmark_pikafish.py` 会读取其中的路径变量。

## 原始数据

输入数据统一经过结构化解析、走法规范化和完整棋局回放。损坏记录不会进入训练数据；来源、棋局编号、结果和走法元数据保留在统一 JSONL 中。

原始棋谱来源：

- [CGLemon/chinese-chess-PGN](https://github.com/CGLemon/chinese-chess-PGN)：中国象棋 PGN 棋谱。
- [weiyinfu/xqp](https://github.com/weiyinfu/xqp)：中国象棋棋谱及相关数据。

使用这些数据前，请分别核实原仓库的授权条款和数据使用范围，并将下载内容放入 `data/raw/`。

统一编码约定：棋盘张量为 `(15, 10, 9)` 的 `float32`，走法起点和终点索引范围为 `0~89`。数据按棋局切分，避免同一棋局同时出现在 train、validation 和 test。

## 统一中间格式

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
  --book-input-dir data\raw\xqp `
  --output data\processed\human_games `
  --shard-size 8192
```

续跑时使用相同输入和 `--resume`：

```powershell
uv run python scripts\unify_format.py `
  --pgn data\raw\dpxq-99813games.pgns data\raw\WXF-41743games.pgns `
  --book-input-dir data\raw\xqp `
  --output data\processed\human_games `
  --shard-size 8192 `
  --resume
```

`--max-games 1 --shard-size 1` 可用于小规模 smoke。统一阶段不调用 Pikafish，也不展开逐 ply 样本。

### 去重与来源

去重键由规范化初始 FEN、完整走法序列和结果组成。不同来源的相同棋局只保留一份，重复记录写入 `duplicates.jsonl`。发布数据集或模型前，应单独核实各来源棋谱和解析工具的许可范围。

## 棋局输赢作为 value

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

### Mirror 数据增强

`train.py --mirror` 在每个训练 batch 内追加左右镜像样本，不生成新的 NPY shard。它沿棋盘宽度翻转 positions，并把每个 start/end 格点从 `9r+c` 映射为 `9r+(8-c)`；value 标签直接复制。validation、test、推理均不镜像。

镜像会使该 batch 的实际前向/反向样本数加倍，因此显存和训练时间也随之增加；`--batch-size` 应按未镜像样本理解。该开关当前只适用于 `train.py` 的 start/end policy 标签。`train_pikafish.py` 尚未实现 `--mirror`，不能在蒸馏训练命令中假设存在该数据增强。

## Pikafish 标注 value

### Pikafish CPU 内置 benchmark

`benchmark_pikafish.py` 仅调用引擎内置 `bench`，不读取任何棋局数据。脚本会自动：

- 从 `.env.local` 读取 `PIKAFISH_PATH` 和 `PIKAFISH_NNUE_PATH`
- 默认扫描 `PIKAFISH_PATH` 同目录下所有 `pikafish-*.exe`
- 对每个引擎执行 `uci -> setoption EvalFile -> isready -> bench -> quit`
- 输出每个引擎的 JSON 结果（`nps`、`totalNodes`、`elapsed`、`exitCode`）
- 输出按 `nps` 排序的 `rankingByNps` 和 `recommendedDefaultEngine`

运行全部可见 CPU 版本：

```powershell
uv run python scripts\benchmark_pikafish.py
```

只测试指定引擎：

```powershell
uv run python scripts\benchmark_pikafish.py --engine C:\workspace\Pikafish.2026-01-02\Windows\pikafish-avx2.exe
```

如果某些二进制与当前 CPU 指令集不兼容（例如 AVX512/VNNI 机型不匹配），脚本会将其标记为 `compatible=false` 并继续测试其他版本。

`annotate_pikafish.py` 只读取统一 JSONL，不直接解析 PGN 或 XQF。它对每个存在人类走法的局面执行 Pikafish MultiPV 搜索，并写入 canonical JSONL。标注阶段不生成 NPY；后续按所需训练配置使用数据准备脚本生成 NPY。

```powershell
uv run python scripts\annotate_pikafish.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\pikafish-d10-m5 `
  --depth 10 `
  --multipv 5 `
  --pikafish-threads 1
```

搜索预算可用 `--depth N`、`--movetime-ms N` 或 `--nodes N` 三选一。`--workers N` 用于并行处理输入 shard 目录；单个 JSONL 文件不支持多 worker。长任务可使用 `--resume`，失败游戏默认跳过，使用 `--retry-failed` 才会重试失败记录。

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

### Pikafish 标注输出

目录中会写入 `annotation_config.json`、`dataset_summary.json`、按输入 shard 命名的 JSONL 和对应的失败事件流。每个 shard 先写入 `.partial.jsonl`，完成后原子改名；每局写入后 fsync。`--resume` 会扫描 JSONL，只接受 ply 从 0 连续且数量与源棋局一致的完整游戏，并移除 partial 中断尾。普通续跑跳过已失败游戏；使用 `--retry-failed` 才会重试尚未成功的失败游戏。

输出目录绑定 schema、输入 shard 的 SHA-256、depth/时间预算、MultiPV 和 PikaFish 线程数。配置不匹配时会拒绝运行，需使用新的 `--output-dir`。

先分别对各 split 跑 1 局 smoke：

```powershell
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\train-000.jsonl --output-dir data\processed\pikafish-smoke-train-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\validation-000.jsonl --output-dir data\processed\pikafish-smoke-validation-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\test-000.jsonl --output-dir data\processed\pikafish-smoke-test-1 --depth 10 --multipv 5
```

确认三次运行零失败、PV 回放和恢复语义正确后，再为每个命令添加 `--resume`。

### Pikafish 蒸馏数据准备

`prepare_pikafish.py` 将**已完成**的 canonical 标注 JSONL 导出为 current-view 蒸馏 NPY。目录输入优先读取完成的 `.jsonl.zst` shard，并跳过 partial 文件：

```powershell
uv run python scripts\prepare_pikafish.py `
  --input-jsonl data\processed\pikafish-d10-m5 `
  --output-dir data\processed\pikafish-distillation `
  --max-candidates 5
```

每个已完成的标注 `jsonl.zst` 独立生成一个同名前缀的 dataset shard，例如 `test-000-000.jsonl.zst` 生成 `dataset/test-000-000-*.npy`。shard 写入 `(15, 10, 9)` 的 `positions`、teacher 分数类型和数值，以及排序去重后的 `candidate_action_ids`/`candidate_scores`。动作 ID 为 `90 * from + to`，黑方局面先转换到 current-view。合法着以紧凑的 `legal_action_ids` 与 `legal_action_offsets` 保存；游戏信息保存在每局一行的 `games.jsonl` 中，每行包含 `game_id`、`sample_start` 和 `sample_end`，不再写逐样本 `metadata`、`game_indices` 或 `plys`。同一局的样本必须在 shard 内连续写入。候选槽位是否有效由 `candidate_action_ids >= 0` 推导，不额外保存 mask。样本准入由蒸馏训练阶段按各 head 的标签类型独立判定。

prepare 按输入标注 shard 恢复：输出前会检查同名前缀的全部文件，完整 shard 直接跳过，未完成或缺失文件则重新生成。进度以 `shards=已完成/总数` 报告；单个标注 shard 内的记录仍会每 5 秒报告一次。

### 蒸馏 NPY 的形状

把一个 shard 想成 `N` 张局面卡片叠在一起。除合法着外，所有数组的第 `i` 项都描述同一张卡片：

```text
第 i 个局面
  positions[i]              (15, 10, 9)  15 个棋子/行棋方通道的 10 x 9 棋盘
  teacher_cp[i]             ()           Pikafish 原始 cp；无效 value 为 NaN
  teacher_score_kinds[i]    ()           0=cp，1=mate
  teacher_scores[i]         ()           teacher 原始分数
  candidate_action_ids[i]   (5,)         Top-5 候选动作，不足 5 个用 -1 补齐
  candidate_score_kinds[i]  (5,)         0=cp，1=mate
  candidate_scores[i]       (5,)         与候选动作一一对应的原始 cp
```

因此常规固定形状字段为：

```text
positions               (N, 15, 10, 9)
teacher_cp              (N,)
teacher_score_kinds     (N,)
teacher_scores          (N,)
candidate_action_ids    (N, 5)
candidate_score_kinds   (N, 5)
candidate_scores        (N, 5)
```

一个联合动作 ID 可拆回棋盘坐标：`from = action_id // 90`、`to = action_id % 90`。例如 `C3-C4` 的两个格点索引为 `29` 和 `38`，其 ID 是 `90 * 29 + 38 = 2648`。candidate 的有效 ID 必须同时出现在同一局面的合法着集合中。

每局合法着数量不同，不能写成低效的 `(N, 8100)` mask。因此所有合法 action ID 拼成一条长数组，再用 offsets 切回各局面：

```text
legal_action_ids      (L,)       [局面 0 的全部合法着 | 局面 1 的全部合法着 | ...]
legal_action_offsets  (N + 1,)   [0, L_0, L_0 + L_1, ..., L]

局面 i 的合法着 = legal_action_ids[offsets[i] : offsets[i + 1]]
```

这里 $L = \sum_i L_i$，$L_i$ 是第 `i` 个局面的合法着数量。这样训练期可只对该切片的 logits 做 softmax，不会为 `8100` 个动作逐局存储布尔掩码。每个 shard 另有 `games.jsonl`：每行保存一局的 `game_id` 及其样本半开区间 `[sample_start, sample_end)`，供审计和 game-level sampling；不再生成逐样本的 `game_indices.npy` 或 `plys.npy`。

### Pikafish 蒸馏训练脚本

`train_pikafish.py` 是 Pikafish joint policy/value 蒸馏的正式训练入口。它读取上述 ragged 合法着 NPY，训练 `PikafishResNet` 的空间式 `8100`-logit policy head 和 bounded value head。

当前没有可用于全量 joint policy/value 训练的推荐命令。`--value-learning-rate` 已提供独立 value-head 参数组；必须先通过 value 稳定化短实验，再将验证后的参数写入本节。

不传 resume 参数时，脚本会自动从 `checkpoint-dir/last.pt` 恢复模型、optimizer、scheduler、AMP scaler 和早停状态。只在每个完整 epoch 结束时保存 checkpoint：刷新 `last.pt`、写入 `epoch-xxxx.pt`，指标改善时写入 `best.pt`。因此进程在 epoch 中途终止时，恢复会从上一个完整 epoch 开始。`checkpoint-dir/training.jsonl` 追加 `training_start`、`resume`、`train_progress`、epoch 汇总和 `early_stop` 事件；progress 包括吞吐、当前进程的 ETA、累计样本数、学习率和 CUDA 显存。`--max-steps` 用于 smoke，但结束前仍会最多跑 50 个 validation batches。

完整 epoch 之间可以改变 `--micro-batch-size`；由于 global batch 保持不变，仍须通过 validation 确认新 micro batch 的数值稳定性。

### Pikafish CUDA 批量探测

`probe_pikafish_batch.py` 用随机位置和 joint policy/value loss 探测显存与纯训练吞吐，不读取蒸馏数据。它输出每个 batch 的参数量、positions/s、peak allocated/reserved GiB、总显存、OOM 状态与保留 10% 显存余量的最大建议 batch。

```powershell
uv run python scripts\probe_pikafish_batch.py `
  --channels 192 --blocks 12 `
  --batch-sizes 1024 1536 2048 4096 `
  --warmup-steps 2 --measure-steps 5
```

该探测不替代真实训练性能测试：真实训练还包含 ragged 合法着、DataLoader、梯度累积、checkpoint 和 validation。正式切换 batch 前应以相同 validation split 做短跑确认。

### Pikafish 蒸馏训练计划（完整记录）

本文定义当前 `schema_version: 1` Pikafish 标注的监督蒸馏目标、数据准入和离线验证。它是训练设计说明，不代表相关 NPY 导出或 loss 已经实现。

#### 已有监督信号

每条样本提供当前局面 `fen`、人类实战 `move`，以及当前行棋方视角的 `teacher`：

- `teacher.score`：Pikafish 对走前局面的原始评估；`cp` 与 `mate` 必须分开处理。
- `teacher.bestmove`：Pikafish 的 Top-1 首着。
- `teacher.candidates`：最多五个 MultiPV 候选及其当前行棋方视角分数。

当前格式没有独立的 `human_score`。`move` 是人类实际选择，不应把 `teacher.score` 错称为该走法的评分。若人类走法命中候选，可读取对应 candidate 的 score；未进入 Top-5 的走法没有精确 score。

所有 score 的符号都必须保持为当前行棋方视角。转换为 current-side-view 时不再额外取反；若改为固定红方视角，才按行棋方转换标签符号。

#### 首轮目标

首轮训练 Pika 策略和 Pika 价值，不将人类 `move` 混入训练 loss：

1. policy target：纯 `cp` candidate 集构造稀疏软标签；任一 candidate 为 `mate` 时改用原始 MultiPV rank 的硬标签。
2. value target：仅由 finite `cp` 的 `teacher.score` 构造有界连续标签。
3. human `move`：仅作数据质量与覆盖率统计。

人类走法统计包括：人类着命中 Pika Top-1/Top-5 的比例；命中时的 best-candidate 分数差；按 ply、局面阶段和 split 的分桶。它用于发现 FEN/走法对齐问题并刻画原棋谱质量，不作为“棋力更强”模型的优化目标。

#### 样本准入与缺失标签

policy 与 value 独立判定样本资格，不因另一 head 缺失标签而丢弃有效监督：

- prepare 只保存原始 candidate 和 root score；policy/value 的样本准入在训练阶段按 score type 和训练配置判定。
- candidate 按输入 MultiPV `rank` 保留，最多五项；rank 是 teacher 的唯一主排序。重复走法保留 rank 最小项；同 rank 冲突视为数据错误。
- `cp`、`mate`、缺失 score、非法 candidate、无法解析 FEN 或候选为空的记录必须可区分统计；不应在 prepare 阶段用 pseudo-cp 覆盖原始值。
- 一个 batch 中分别对有效 policy 与 value 样本求均值；某 head 在该 batch 没有有效样本时，其 loss 为零且不参与该 head 的分母。

训练、validation、test 只使用完整的非 partial shard，并按 `game_id` 保持既有 split；相同 `game_id` 不得跨 split。

去重后的 rank 最小 candidate 是 PV1；`teacher.bestmove` 应与它一致。不一致时记录审计错误，但 policy 标签以 candidate 集为准，不因该冗余字段丢弃样本。

#### 动作空间与合法着掩码

首轮 policy 使用起点和终点的联合分类空间。current-view 的每个格点使用行优先索引 $i=9r+c$，其中 $0\leq r<10$、$0\leq c<9$；动作 ID 为：

$$
\operatorname{id}(a)=90\cdot\operatorname{from}(a)+\operatorname{to}(a)
$$

policy head 固定输出 `90 * 90 = 8100` 个联合动作 logits。实现使用空间卷积 head：输出张量为 `[batch, 90, 10, 9]`，空间位置表示起点、channel 表示终点；按上述 `id(from, to)` 展平。它直接建模每个 `(from, to)` 配对，不使用独立 start/end head。每个 candidate、`bestmove` 审计项和棋规生成的合法着均须先转换为该 ID；不另设“无着”类别。训练与评估均从棋规引擎生成 $\mathcal{L}(s)$，对不在该集合内的 logits 置为 $-\infty$ 后再计算 softmax。终局局面没有 policy loss；非终局局面若合法着为空视为数据错误。

#### Policy 蒸馏

纯 cp policy slice 要求所有保留 candidate 都有有效 cp score。对候选 cp score `s_i`，使用相对分数避免数值溢出：

$$
q_i = \frac{\exp((s_i-s_1)/\tau)}{\sum_j \exp((s_j-s_1)/\tau)}
$$

其中 `s_1` 是 rank 最小 candidate 的 score，`tau` 单位为 cp。令 $C(s)$ 为有效 candidate 集，$\mathcal{L}(s)$ 为棋规生成的全部合法着，模型 logits 为 $z_\theta(s,a)$，则：

$$
P_\theta(a\mid s)=
\frac{\exp(z_\theta(s,a))\mathbf{1}[a\in\mathcal{L}(s)]}
{\sum_{b\in\mathcal{L}(s)}\exp(z_\theta(s,b))}
$$

$$
L_{\mathrm{policy}}(s)=-\sum_{a\in C(s)}q(a\mid s)\log P_\theta(a\mid s)
$$

训练时只对合法着归一化，并以稀疏 $q$ 计算交叉熵；Top-5 外的合法着不是非法走法，也不应从 softmax 分母移除。`teacher.bestmove` 仅作与 rank-PV1 的一致性审计，不单独构造标签。

任一 candidate 为 mate 时，该局面进入 mate policy slice，不与 cp softmax 或温度混用。以 rank 最小 candidate 为唯一硬标签，因而同时覆盖最短成杀、唯一防杀和延缓被杀。该样本不参与 value loss。mate slice 的逐样本 policy loss 权重为 $4$，cp slice 权重为 $1$：

$$
L_{\mathrm{policy}}=
\frac{\sum_i w_i\ell_i}{\sum_i w_i},
\qquad
w_i=\begin{cases}
4,&\text{mate slice}\\
1,&\text{cp slice}
\end{cases}
$$

若任一保留 candidate 缺失 score、score type 无效、走法非法或无法映射动作 ID，整个局面不参与 policy loss，并记录排除原因；其 value 标签仍可独立使用。

当前完成 shard 的观测值中，最佳与第二候选分差为 P50 `11 cp`、P75 `35 cp`、P90 `98 cp`、P95 `183 cp`。首轮测试：

```text
tau = 25, 50, 100 cp
```

`50 cp` 是 cp-only candidate 的首个基线：候选接近时保持软分布，明显优势时才显著偏向 PV1。含 mate candidate 的局面使用独立 hard-label mate slice，不覆盖原始 score。

#### Current-view 与镜像增强

训练和推理默认使用 current-view：黑方行棋时先旋转棋盘 $180^\circ$、交换双方棋子通道，并转换走法坐标，使当前行棋方始终处于同一侧。policy 和 value 因而都以当前行棋方视角定义。

在 current-view 归一化后，训练集额外使用左右镜像增强。对棋盘宽度 `9`，任一方格索引 `i = 9r + c` 的镜像为：

$$
M(i) = 9r + (8-c)
$$

镜像规则如下：

- 每个训练 batch 可独立应用 `--mirror`；validation、test 和推理不镜像。
- policy 的起点、终点及所有 MultiPV candidate 走法必须应用 $M$；soft-policy 的 candidate score 与概率不变。
- value 是同一局面的当前行棋方评估，镜像前后不变。
- 合法着集合也必须映射到镜像坐标；Top-5 外的合法着仍保持合法、可探索。

变换顺序固定为：`绝对局面 -> current-view -> 可选左右镜像`。不要以红方视角镜像后再猜测黑方标签的符号；current-view 已统一 value 的符号，镜像只改变左右坐标。

该组合同时利用两种不同对称性：current-view 共享红黑双方的前后对称，左右镜像共享九路棋盘的左右对称。它们不重复，也不会改变将帅、兵卒或车马炮的合法走法规则。

#### Value 蒸馏

`teacher.score` 是 Pika 搜索评估，不是已经标定的胜率或终局期望结果。若实际 Pikafish 二进制提供 `UCI_ShowWDL`，其 WDL 也只能按 Pikafish 自己的模型解释，不能搬用国际象棋 Stockfish 的 cp-to-WDL 公式。

当前完成 shard 的 cp 分布如下：

| 指标 | 值 |
|---|---:|
| 样本数 | 6,561 |
| mate 样本 | 51 / 6,612 |
| `abs(cp)` P50 / P75 / P90 | 51 / 162 / 319 |
| `abs(cp)` P95 / P99 / 最大值 | 399 / 556 / 900 |

为让训练目标和模型输出具有稳定量纲，首轮采用：

$$
V(s) = \tanh(\operatorname{clip}(\mathrm{cp}, -900, 900) / K)
$$

其中 `K=400` 是首个候选，因为它约等于当前 shard 的 `abs(cp)` P95；它只是压缩尺度，不是已标定的胜率，也不是通用常数。模型 value head 最后一层必须同步添加 `tanh`，以 Huber loss（$\delta=0.1$）拟合该标签；MSE 仅作为对照指标，不作为首个优化目标。mate root score 不构造 value label。

首轮扫描：

```text
K = 300, 400, 600
```

`K=400` 时，当前 shard 的 `abs(cp)` P50/P90/P95/P99 约映射为 `0.127/0.663/0.760/0.883`；只有少数极端局面接近 `+1` 或 `-1`。该范围能限制异常 cp 对回归梯度的支配，同时保留常见局面的分辨率。

#### 训练与验证

在已完成的非 partial shard 上，按 `game_id` 保持既有 train/validation/test split。第一阶段不等待全量标注完成，但只运行短实验：

##### 数据切分执行状态（2026-08-25）

`data/processed/pikafish-distillation/dataset` 已完成一次仅重命名不重导出的 split 收缩，规则与结果如下：

- 保留 validation 与 test 各前 11 个 shard 不变；
- 其余 shard 按前缀组整体迁移到 train：`test-000 -> train-012`、`test-001 -> train-013`、`validation-000 -> train-014`、`validation-001 -> train-015`；
- 映射清单见 `data/processed/pikafish-distillation/remap_plan_1pct.csv`；
- 迁移后统计（见 `data/processed/pikafish-distillation/dataset_summary.json`）：
	- train: 1100 shards, 8,987,007 samples
	- validation: 11 shards, 90,570 samples
	- test: 11 shards, 90,687 samples

后续实验固定使用该 split，不再回退到原 8:1:1 切分。

```text
value scale K:        300, 400, 600
policy temperature:   25, 50, 100 cp (cp slice only)
policy/value weight:  1:1, 1:0.5, 1:2
mate handling:        policy hard label, weight 4; excluded from value
human move loss:      disabled
value output:          tanh, bounded to [-1, 1]
```

联合目标为：

$$
L=\lambda_pL_{\mathrm{policy}}+\lambda_vL_{\mathrm{value}}
$$

其中两个子 loss 均先在各自有效样本上求均值，再施加权重；禁止以无效样本补零后参与均值。

##### 采样与聚合

训练每个 epoch 无重复、无遗漏地遍历 train split 的全部局面；长棋局按其局面数拥有更高权重。镜像在抽样后独立以 `0.5` 概率应用。

validation 与 test 不重采样、不开镜像，遍历各 split 的全部有效局面。主要 loss、Top-1/Top-5 和 MAE 按有效局面 micro-average；同时报告按 `game_id` 先聚合再平均的 macro-average，以检查长局偏置。

##### 数据顺序与全局打乱

全局索引映射（为所有样本建立前缀和，再用 sampler 生成全局索引）在 map-style dataset 上是正确的：一轮可以无重复、无遗漏地访问全部样本。当前策略将其作为统一索引层，而非默认“逐样本全局随机读盘”策略。

Pikafish 训练采用 map-style dataset 作为正式接口：dataset 的 `__len__()` 返回 split 的总样本数，`__getitem__(global_index)` 通过 shard 累计长度将全局索引映射为 `(shard_id, local_index)`，并按局部 offsets 读取该样本的合法着。

在当前数据规模（train 约 9M 样本、1100 shards）下，训练侧采用分层策略：

- 索引层：统一使用 index mapping，保证可复现与无重复/无遗漏；
- 采样层（生产默认）：block shuffle，而非逐样本全局随机；
- 采样层（小规模 debug）：可启用精确全局 `randperm`，用于正确性与基线对照。

生产模式的 block shuffle 采用“块间随机 + 块内随机”：

- 先按全局索引切分固定大小 block；
- 每个 epoch 随机 block 顺序；
- 对每个 block 的局部索引做随机 permutation；
- 最终按 index mapping 回到 `(shard_id, local_index)` 读取样本。

首轮默认 `block_size = 64K`，并以 `32K` 作为主要对照。选择标准：在验证质量不恶化的前提下优先吞吐更高配置。

固定约束如下：

- 必须按完整 `game_id` 隔离 train/validation/test，不能为了全局打乱而重新按局面切分；
- 评估集保持固定顺序，确保不同实验可比较；
- 多 worker 使用同一 epoch 随机序列（按块切分工作），确保全局无重复、无遗漏；
- 断点恢复需保存 epoch 种子、块游标与块内游标，恢复后不重洗牌。

验收与切换门槛：

- 对照项：A=shard shuffle（基线），B=block 32K，C=block 64K；
- 核心指标：samples/s、GPU 利用率、单 epoch 时长、validation 曲线方差；
- 通过条件（相对 A）：吞吐下降不超过 5%，validation 方差恶化不超过 10%；
- 若 B/C 同时达标，选择吞吐更高者；若 64K 方差超阈值，则回退到 32K。

该方案不要求修改 `prepare_pikafish.py` 的 NPY 输出。训练侧只需维护 shard manifest、累计样本数、block 边界与随机状态；`legal_action_offsets` 仍按 shard 内局部索引读取，并在 batch 级保留 flat action IDs 与 batch offsets。

##### 架构与固定训练配方

首轮冻结架构为 `PikafishResNet(channels=192, blocks=12)`，实现见 [src/backend/models/pikafish_resnet.py](../src/backend/models/pikafish_resnet.py)，输入与 current-view 编码沿用现有 ResNet。它独立于旧 policy baseline 的 start/end head，并提供上文定义的空间式 8100-logit 联合动作 head 和有界 value head。架构、输入通道和 action-head 布局在本实验族内不可变；任何架构调整另开实验族。

在正式训练前，使用该固定架构进行短显存/吞吐探测。训练流程分为两阶段：先完成小 batch smoke 建立指标口径，再迁移到大 batch 吞吐轨并验证主干/policy 与 value head 的独立学习率和 warmup。

2026-08-25 已在 RTX 2060 Super 8GB 上以真实 policy 交叉熵、Huber value loss 和 AdamW 更新探测 `192x12`：早期 smoke 在 `micro batch=64` 下无 OOM，peak allocated/reserved 为 `0.273/0.297 GiB`，吞吐约 `2,042 positions/s`。随后完成大 batch 轨迁移（见下文“参数锁定与数值评估更新”）。探测脚本为 [scripts/probe_pikafish_batch.py](../scripts/probe_pikafish_batch.py)。

除实验矩阵中的 `K`、`tau` 和 policy/value 权重外，当前执行基线如下：

```text
optimizer:            AdamW
micro/global batch:   2048 / 4096 positions
maximum epochs:       100
early stopping:       validation J_select, patience 5 epochs, min_delta 0.004
random seeds:         3 fixed seeds per matrix point
checkpoint selection: lowest validation J_select subject to slice gates
```

##### 单机硬件参考配置（RTX 2060 Super 8GB, RAM 32GB）

该硬件采用“大 batch 稳态默认 + 上限探测回退”策略：日常以留余量稳态配置运行，4096/4096 仅用于上限探测与短压测。

当前固定轨（默认执行）：

- micro/global batch 固定为 `2048/4096`；
- 当前学习率候选为 main/policy `4e-4`、value head `1e-5`；它只通过 50-update smoke，尚未锁定为全量训练参数；
- scheduler、weight decay、policy/value weight 与 value scale 均须在 value 稳定化实验中重新锁定；
- AMP 保持开启，作为 8GB 显存下的默认配置；
- dataloader workers 固定为 8（单进程上限），prefetch factor 固定为 4。

上限探测与回退轨（仅对照）：

- 上限探测：`micro/global=4096/4096`（观测显存约 `7794 MiB`，GPU 利用率 `100%`）；
- 日常默认：`micro/global=2048/4096`；
- 若出现偶发 OOM 或显存碎片抖动，回退到 `micro/global=1536/4096`；
- 选择依据仍以 validation 指标稳定性为主，不以瞬时吞吐为主。

若需要在 1536/2048/4096 三档间切换，必须在相同 validation/test split 下对照，并额外记录：

- validation 曲线方差；
- 到达同等 validation loss 的步数；
- 最终 checkpoint 的 test 指标。

global/micro batch 改变时，学习率不按线性规则放大，必须与 warmup 一起重新验证。当前候选保持 global batch `4096`，主干/policy base LR 为 `4e-4`，value head base LR 为 `1e-5`。每个完整 matrix point 的三个 seed 分别训练和验证，选择时比较 validation 指标的均值与标准差；三个 seed 的最终 checkpoint 都运行 test，报告均值与标准差。test 不参与早停或参数选择。

记录以下指标：

- 数据：各 split 的游戏数、局面数、policy/value 有效数、排除原因计数，以及 ply 与局面阶段分桶分布；
- policy：cp slice 的 candidate KL、Pika bestmove Top-1、candidate Top-5 和候选概率质量；mate slice 的 hard-label Top-1；
- value：有界 target MAE、Pearson/Spearman、按 $|V|$ 分桶的 MAE、符号正确率与输出饱和比例；另报告 $|cp|\le300$ 的原始 cp-MAE；
- 人类审计：人类 move 的 Pika Top-1/Top-5 命中率；
- 对称性：镜像前后 candidate 概率与 value 的差异；current-view 对换前后的 value 是否变号。

选择方案时，以固定 validation/test split 的蒸馏指标为准。不能直接比较不同 `K` 下的归一化 MSE，也不能将不同 $\tau$ 的原始交叉熵直接比较。固定基线为 `K=400, tau=50, policy:value=1:1`；定义：

$$
J_{\mathrm{select}}=
\frac{1}{2}\frac{KL}{KL_{\mathrm{base}}}+
\frac{1}{2}\frac{MAE_{\mathrm{cp},|cp|\le300}}{MAE_{\mathrm{cp},\mathrm{base},|cp|\le300}}
$$

阶段一每个组合以一个固定 seed 训练 3 epochs，按 $J_{\mathrm{select}}$ 取前三名。阶段二仅让这三个组合各训练三个固定 seed，并以三个 seed 的 validation 均值选择胜者；均值接近时优先标准差更小者。所有选择还须满足：cp-policy、cp-value 和 mate-policy 均不劣于当前最佳值的预设门槛；mate-policy 使用 hard-label Top-1，最多下降 2 个百分点。若 validation mate 样本少于 500，该门槛只告警不阻止选择。

##### Smoke 结果（2026-08-25）

以固定 seed、`192x12`、micro batch `64`、global batch `256` 和 200 个 optimizer updates 对 50 个 validation batch 做 smoke。所有组合均无 OOM，且 policy/value/mate 有效标签计数一致。跨 $K$ 的比较使用 $|cp|\le300$ 原始 cp-MAE；跨 $\tau$ 的比较使用 cp-policy KL。

| 配置 | cp-policy KL | $\lvert cp \rvert\le300$ cp-MAE |
|---|---:|---:|
| `K=300, tau=100, 1:0.5` | **1.9333** | **59.08** |
| `K=300, tau=100, 1:1` | 1.9372 | 62.42 |
| `K=300, tau=100, 1:2` | 1.9572 | 65.61 |
| `K=300, tau=25, 1:1` | 2.2052 | 60.76 |
| `K=400, tau=50, 1:1` | 2.0510 | 62.55 |
| `K=600, tau=25, 1:1` | 约 2.21 | 69.00 |

结论：

- `K=300, tau=100, policy:value=1:0.5` 在该短跑中同时最好；完成 value 稳定化后才可重新验证它是否适合作为初筛候选；
- `1:2` 明显损害 cp-policy KL 和原始 cp-MAE，暂不保留更高 value 权重；
- `K=600` 的低有界 value loss 是尺度效应，原始 cp-MAE 明显更差，淘汰；
- `tau=100` 的 cp-policy KL 最低，优于已测的 `tau=25/50`；
- `K=400, tau=50, 1:1` 仍保留为 $J_{\mathrm{select}}$ 的固定归一化基线，不作为默认训练参数。

##### Value 饱和失效实验（2026-08-25）

`192x12`、`K=300`、`tau=100`、policy:value=`1:0.5`、`micro/global=2048/4096` 的一次完整 epoch 使用 base learning rate `1.2e-3` 后，policy 学到了 teacher 排序，但 value head 完全失效：

| 指标 | 结果 |
|---|---:|
| cp-policy KL | `1.0701` |
| $|cp|\le300$ cp-MAE | `1139.996` |
| validation $J_{\mathrm{select}}$ | `9.3736` |
| 训练 update 墙钟 | `2126.9s`，约 `35分27秒` |

完整 validation 的 value 输出在 float32 和 AMP float16 下均为 `+1.0`；`tanh` 前激活范围为 `[20.2, 70.6]`，因此不是 AMP 显示误差。该 checkpoint 不可发布、不可用于对弈评估、不可作为正式训练续点。

最小诊断排除了 policy 干扰和明显的样本标签偏置：只保留 value loss 时也在 `1.2e-3` 的第一个 optimizer update 饱和；首个训练 batch 的 bounded cp target 均值仅 `+0.0207`，范围为 `[-0.978, +0.986]`。单次 value-only update 的学习率扫描如下：

| learning rate | 更新后 value 均值 | `tanh` 前均值 | 判定 |
|---:|---:|---:|---|
| `1e-5` | `0.022` | `0.022` | 稳定 |
| `4e-5` | `0.217` | `0.220` | 未饱和 |
| `1e-4` | `0.470` | `0.511` | 明显漂移 |
| `4e-4` | `0.957` | `1.940` | 接近饱和 |
| `1.2e-3` | 约 `1.000` | `6.48` | 单步完全饱和 |

根因是当前 value head 的 `Flatten(192*10*9) -> Linear(...,128) -> ReLU -> Linear(128,1) -> Tanh` 在 Adam 高学习率首步下发生高维累积更新，迅速把 pre-tanh 推入饱和区。常规 global-norm clipping 不能作为首要修复，因为 Adam 首步对统一缩放的梯度近似保持相同的更新方向和量级。

后续必须新开 checkpoint 目录，先验证稳定化实验，再重新锁定配方：

1. value head 使用独立、明显更低的学习率 `1e-5`；主干/policy 使用候选 `4e-4`，两者仍须跨 warmup 单独测量。
2. 结构性候选是先空间池化再做小 MLP，避免高维 flatten value head。
3. 每个候选先跑 `20/100/200` updates，记录 value 的 pre-tanh、`abs(value)>=0.99` 比例、原始 cp-MAE 和 policy KL；所有 value 输出保持非饱和后才允许跑完整 epoch。

已实现的候选参数组为 `main/policy=4e-4`、`value_head=1e-5`。真实 `50`-update smoke（`micro/global=2048/4096`、workers=`8`）得到 cp-policy KL=`1.7247`、value cp-MAE=`69.80`、$J_{\mathrm{select}}=0.9784`，没有出现饱和。当前学习率调度将两个参数组乘以相同的 warmup/cosine scale：主干的末端 LR 是 `1e-5`，value head 的末端 LR 为 $1e-5\times(1e-5/4e-4)=2.5e-7$。候选使用固定 `110` 个 warmup steps，避免 `--epochs` 改变时按比例 warmup 变成多个 epoch。该结果只证明早期稳定性；跨 warmup 的 `100/200`-update 复测仍是放行全量训练的前置条件。

候选短 smoke 命令：

```powershell
uv run python scripts\train_pikafish.py `
  --epochs 1 --max-steps 50 `
  --learning-rate 4e-4 --value-learning-rate 1e-5 --min-learning-rate 1e-5 `
  --warmup-steps 110 --weight-decay 1e-4 `
  --temperature 100 --value-scale 300 --policy-weight 1 --value-weight 0.5 `
  --micro-batch-size 2048 --global-batch-size 4096 `
  --num-workers 8 --prefetch-factor 4 --seed 42 `
  --checkpoint-dir checkpoints\pikafish-lr4e4-vlr1e5-s50
```

#### 当前执行状态

当前没有已验证的全量 joint policy/value 执行配方。必须先完成上方 value 稳定化实验；在此之前仅可复现 policy 吞吐或做受控短诊断，不能将任何新 checkpoint 视为可发布模型。

下表保留的是已验证的硬件吞吐档位，**不是** value 已稳定的训练参数：

| 项目 | 日常吞吐档 | 上限探测 | 回退档（不稳时） |
|---|---|---|---|
| micro/global batch | `2048 / 4096` | `4096 / 4096` | `1536 / 4096` |
| workers / prefetch | `8 / 4` | `8 / 4` | `8 / 4` |

样本量换算规则（防止“大 batch 看起来更慢”）：

- 等价步数换算：`new_steps = old_steps * old_global_batch / new_global_batch`。
- 例：`10000 steps @ global=128` 约等价 `313 steps @ global=4096`。

batch 实测更新（2026-08-25）：

- 在 `micro/global=4096/4096` 下，单卡 RTX 2060 Super 8GB 观测到显存占用约 `7794 MiB`、GPU 利用率 `100%`；
- 该配置用于上限探测与短压测，不作为日常默认；
- 日常默认改为留余量稳态：`micro/global=2048/4096`；
- 若仍有偶发 OOM 或显存碎片抖动，继续回退到 `micro/global=1536/4096`。
- 大 batch 的吞吐测量保持有效。当前只允许使用 main/policy `4e-4` 与 value head `1e-5` 做短 smoke；全量训练学习率尚未放行。

#### 性能优化实测（2026-08-25）

以下吞吐数据均为 `PikafishResNet(192x12)`、CUDA、AMP、AdamW、`num_workers=8`、`prefetch_factor=4` 的短训练实测。Windows 下首个 update 包含 DataLoader worker spawn，不能用于稳态吞吐比较；统一取同一进程的第二个 optimizer update。`--max-steps` 结束时仍会执行 validation，因此 epoch 总时长也不用于本表。

| 配置 | 稳态 update 时间 | 相对 `2048/4096` | 结论 |
|---|---:|---:|---|
| `micro/global=2048/4096` | `1.038 s` | 基准 | 日常默认，保留显存余量 |
| `micro/global=4096/4096` | `0.820 s` | `1.27x`（约快 `21%`） | 可用于短压测；接近 8GB 显存上限 |

优化过程与结果：

- 首次 `torch.utils.bottleneck` 显示 Python 端主要热点为逐行合法着掩码、逐行 candidate 合法性检查及逐行 cp policy loss；`compute_losses` self time 为 `13.456 s`，`_masked_log_probs` 为 `4.755 s`，`torch.isin` 为 `5.287 s`。
- 训练脚本已将以上逻辑改为批量合法着 mask、批量 `gather` 和批量 softmax。相同的微型 profiler 采样下，cProfile 总时间由 `37.078 s` 降至 `17.452 s`，`compute_losses` self time 降至 `1.821 s`；loss 语义由训练单测覆盖。
- 向量化后，真实大 batch 的稳态路径不再受数据读取限制：`2048/4096` 连续第 2 个 update 为 `1.038 s`。不为减少 NPY shard 的启动期 `open/stat/memmap` 开销引入额外缓存，因为它不改善长训练吞吐。
- 已测试 `torch.compile`。当前 Windows + PyTorch `2.6.0+cu124` 环境的 Inductor 无可用 Triton backend，报错 `Cannot find a working triton installation`，因此当前训练脚本保持 eager 模式。后续仅在具备可用 Triton 的 Linux/WSL2 CUDA 环境重新评估 compile；不将其列为当前训练依赖。

切换 `2048/4096` 与 `4096/4096` 时保持 global batch 都为 `4096`，但仍须以相同 seed 和完整 validation 曲线确认显存余量与 value 稳定性。当前候选学习率和 warmup 仅用于 `2048/4096`；切到 `4096/4096` 时必须重新验证，不能直接复用。

1 epoch 耗时估算（先验）：

- 当前配置下每 epoch 更新步数约为 `2195`；
- 估算公式：`epoch_seconds ≈ 2195 * avg_step_seconds`；
- 参考区间：
	- `avg_step_seconds=1.038` 时约 `38` 分钟（`2048/4096` 短测稳态值）；
	- `avg_step_seconds=0.820` 时约 `30` 分钟（`4096/4096` 短测稳态值）；
	- 以上只估算训练 update，不含每个 epoch 的完整 validation、checkpoint 写入和偶发 DataLoader 抖动。

建议先运行 30~50 steps，读取日志中的 `avg_step_seconds` 后再代入上式，得到本机当次训练更准确的 1 epoch 预估。

#### 实现顺序

1. 新增 Pikafish JSONL 到 NPY 的导出器，保留原始 cp、候选、有效性标记和排除原因。
2. 扩展数据加载和训练循环，支持稀疏 candidate soft-policy loss、独立 head mask 和联合损失权重。
3. 为样本准入、有界 value label、mate 过滤、current-view、左右镜像 candidate 变换和 game-level split 添加测试。
4. 用小型固定 shard 跑实验矩阵，确定 `K`、`tau` 与 policy/value 权重后再导出全部已标注数据。
5. 标注覆盖率提高后，以相同 validation/test split 扩容复训；不要把 partial shard 作为训练输入。
6. 新增独立的 Pikafish map-style 训练脚本，先用小型固定数据验证全局索引、worker 无重复、跨 shard 边界和 ragged offsets；不修改 ResNet baseline 的训练接口。
7. 在扩大 batch 或增加 shard 后，基准测试精确全局 `randperm` 与 block shuffle 的吞吐、GPU 利用率、内存峰值和 validation 方差；生产训练不默认采用 10M 级全局 `randperm`。


### Pikafish `cp` 与 `mate` 的统一方式（完整研究记录）

本笔记核对 Pikafish 与 Stockfish 当前官方 `master` 源码，回答训练数据能否将 UCI `score cp` 与 `score mate` 合成一个连续分数。结论适用于 2026-08-21 获取的源码；本地二进制可能来自较早 commit，应以其 `uci` 握手输出为准。

#### 结论

两种引擎都在**搜索内部**使用单个可排序整数 `Value`，但在 **UCI 输出边界**刻意把普通评估和将死评估分成 `cp` 与 `mate` 两种类型。没有官方定义的 cp-to-mate 换算。

因此：

- 引擎搜索可以比较 `cp` 与 mate，因为内部 `Value` 有特殊的 mate 区间；
- 训练数据不应把 UCI `mate 3` 写成“某个很大的 cp”；
- 只训练 exact `cp` 是语义最保守的对照基线；当前项目的首轮配置则让 mate 仅进入独立 policy hard-label slice，不进入 value；
- 若要使用 mate，必须定义并记录新的**训练效用**或独立标签，不能把它称为 Pikafish cp。

#### 引擎内部的统一排序

Pikafish 和 Stockfish 的 [`types.h`](https://github.com/official-pikafish/Pikafish/blob/master/src/types.h) 都定义：

```cpp
constexpr int MAX_PLY = 246;
constexpr Value VALUE_MATE = 32000;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;

constexpr Value mate_in(int ply)  { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }
```

于是搜索内部有：

$$
v_{\mathrm{win}}=32000-p,
\qquad
v_{\mathrm{loss}}=-32000+p
$$

其中 $p$ 是距离终局的 ply 数。较快取胜的值更大；较晚失败的值也更大。这是 alpha-beta、置换表和走法排序需要的**内部序关系**，不是 centipawn 的物理或概率含义。

Pikafish 的 [`score.cpp`](https://github.com/official-pikafish/Pikafish/blob/master/src/score.cpp) 随后按 `is_decisive(v)` 分支：普通值变为 `InternalUnits{UCIEngine::to_cp(v, pos)}`，决定性值变为 `Mate{VALUE_MATE - abs(v)}`。Stockfish 的 [`score.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/score.cpp) 同样处理 mate，但还为 tablebase 保留额外分支。

#### UCI 为什么不把它们合并

Pikafish 的 [`UCIEngine::format_score`](https://github.com/official-pikafish/Pikafish/blob/master/src/uci.cpp) 对两个类型分别输出：

```text
InternalUnits -> score cp <x>
Mate          -> score mate <m>
```

内部 `Mate.plies` 会转换为 UCI 的步数：

$$
m=\begin{cases}
(p+1)/2, & p>0\\
p/2, & p<0
\end{cases}
$$

所以 UCI `mate 3` 表示“当前行棋方有强制将死，按 UCI 约定折算为 3 步”，不是 `300 cp`、`3000 cp` 或内部 `31995`。

同一文件的 [`UCIEngine::to_cp`](https://github.com/official-pikafish/Pikafish/blob/master/src/uci.cpp) 注释明确写着它是在“不处理 mate 与类似特殊分数”的前提下，将普通内部 `Value` 通过当前局面的 material-conditioned 参数转换为显示 cp：

$$
\operatorname{cp}(v,s)=\operatorname{round}\left(\frac{100v}{a(s)}\right)
$$

其中 $a(s)$ 随局面材料变化。这也说明即使普通 UCI cp 也不是“固定内部整数比例”，更不能由 cp 推出将死步数。

#### Pikafish 与 Stockfish 的差异

| 项目 | Pikafish 当前 `master` | Stockfish 当前 `master` |
|---|---|---|
| mate 内部编码 | `VALUE_MATE=32000`，与 Stockfish 相同 | `VALUE_MATE=32000` |
| 普通分数 UCI 输出 | 位置材料相关的 `to_cp()` | 位置材料相关的 `to_cp()` |
| mate UCI 输出 | `score mate <步数>` | `score mate <步数>` |
| tablebase 特殊值 | 当前 `Score` 没有该输出分支 | `Tablebase` 会格式化成特殊的 `cp ±(20000 - plies)` |
| WDL | 当前源码提供 `UCI_ShowWDL` | 提供 `UCI_ShowWDL` |

Stockfish 的 tablebase `cp ±(20000 - plies)` 是协议兼容用的特殊编码，见 [`format_score`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp)；它不能与普通 cp 混合训练。

当前 Pikafish [`Engine::Engine`](https://github.com/official-pikafish/Pikafish/blob/master/src/engine.cpp) 已注册 `UCI_ShowWDL`，且 [`UCIEngine::wdl`](https://github.com/official-pikafish/Pikafish/blob/master/src/uci.cpp) 使用自己的象棋、材料相关 WDL 模型。仓库现有训练计划中“Pikafish 主线没有 `UCI_ShowWDL`”的表述已不适用于当前 `master`；是否可在本项目使用，仍须以实际标注二进制的 `uci` 输出验证。

#### 训练建议

##### 基线：保持 cp 与 mate 分离

若只需最保守的纯 cp 基线，可采用：

```text
value：只使用 teacher.score 为 finite cp 的样本
policy：只使用所有保留 candidate 都为 finite cp 的样本
mate：保留原始 tagged score 与排除统计，但不进入 loss
```

这避免温度 $\tau$、cp clip 与 `tanh` 分母被任意伪 cp 常数支配。当前项目选择比该保守基线多走一步：mate 不进入 value，但进入独立 policy slice，以原始 MultiPV rank 最小 candidate 为硬标签；它不与 cp softmax 混合，也不把 mate 重编码为 pseudo-cp。

##### 必须使用 mate 时

不要复用 `cp` 字段。可选方案如下。

**Policy，保守方案：** 将含 mate candidate 的局面作为单独 slice。若存在正 mate，只训练最短正 mate 的硬标签或 ordinal 排序；全 cp candidate 仍使用原有 softmax。这样两个温度不混用。

**Policy，单一 softmax：** 定义并审计训练效用 $u$，而不是伪装成 cp。设 $C>0$ 是 cp clip，$\delta>0$ 是 mate-distance tie-break：

$$
u(\mathrm{cp})=\operatorname{clip}(\mathrm{cp},-C,C)
$$

$$
u(\mathrm{mate}\ m)=
\begin{cases}
C+\delta/(m+1), & m>0\\
-C-\delta/(|m|+1), & m<0
\end{cases}
$$

然后在独立的 $\tau_u$ 上使用 $\operatorname{softmax}(u/\tau_u)$。这保证任意强制胜排在 clipped cp 之上、任意强制负排在其下，并保留“更快赢、更晚输”的顺序；但 $C$、$\delta$、$\tau_u$ 都是训练超参数，不是引擎数值。

**Value：** 若单一 bounded head 必须吸收 mate，可定义：

$$
y(s)=
\begin{cases}
\tanh(\operatorname{clip}(\mathrm{cp},-C,C)/K), & \mathrm{cp}\\
+1, & \mathrm{mate},\ m>0\\
-1, & \mathrm{mate},\ m<0
\end{cases}
$$

该定义的语义是“有界优势/决定性结果”，不再声称是 cp 回归或校准胜率，并且会丢失 mate distance。若距离本身重要，应添加独立的 `forced_result` 或 `mate_distance` 辅助目标。

**WDL 备选：** 若实际 Pikafish 二进制提供 `UCI_ShowWDL`，可保存 `w d l` 并对非 mate 使用：

$$
y_{\mathrm{wdl}}=(w-l)/1000
$$

mate 设为 $\pm1$。这比 cp-to-WDL 伪公式更可追溯，但它是 Pikafish 自己的象棋 WDL 模型，不能套用 Stockfish 国际象棋参数。

#### 建议的数据表示

score 应保持 tagged union，而不是单个 float：

```text
score_kind: cp | mate
cp:          finite integer | null
mate_moves:  signed integer | null
perspective: side_to_move
```

每个 MultiPV candidate 独立携带该类型。若未来采集 UCI 的 `lowerbound` 或 `upperbound`，也应另存 bound；这种截断搜索界不应默认作为精确 value 回归标签。

## 其它功能脚本

### CUDA 检查

```powershell
uv run python scripts\check_cuda.py
```

### 模型对弈 Benchmark

`benchmark_models.py` 让两个完整 checkpoint 直接进行多盘中国象棋对弈，不依赖前端。它会自动识别 `ResNet` 与 `PikafishResNet` checkpoint，因此可对比同类模型，也可混合对弈。默认对弈 100 盘、交换红黑方，结果写入 `benchmark/model_benchmark.json`。

```powershell
uv run python scripts\benchmark_models.py `
  --model-a models\model-a.pt `
  --model-b models\model-b.pt
```

使用 `--games N` 调整盘数、`--output PATH` 指定结果位置；`--mcts-time SECONDS` 可为带 value head 的 `ResNet` 或 `PikafishResNet` 启用 MCTS。`--fen` 固定开局局面，`--same-colors` 禁止换色。
