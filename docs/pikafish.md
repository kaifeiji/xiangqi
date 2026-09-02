# Pikafish 标注、蒸馏与实验记录

本文整理 Pikafish 相关流程：CPU 引擎 benchmark、MultiPV 标注、蒸馏 NPY、joint policy/value 训练，以及 cp/mate 研究结论。所有命令从仓库根目录执行。

## 环境变量

复制 `.env.example` 为 `.env.local`，或显式设置：

```env
PIKAFISH_PATH=C:\workspace\Pikafish.2026-01-02\Windows\pikafish-avx2.exe
PIKAFISH_NNUE_PATH=C:\workspace\Pikafish.2026-01-02\pikafish.nnue
```

## CPU 引擎 Benchmark

`benchmark_pikafish.py` 调用引擎内置 `bench`，用于选择本机最快且兼容的 Pikafish 二进制，不读取棋局数据。

```powershell
uv run python scripts\benchmark_pikafish.py
uv run python scripts\benchmark_pikafish.py --engine C:\workspace\Pikafish.2026-01-02\Windows\pikafish-avx2.exe
```

输出包含 `nps`、`totalNodes`、`elapsed`、`exitCode`、`rankingByNps` 和 `recommendedDefaultEngine`。CPU 指令集不兼容的二进制会被标记为 `compatible=false`。

## MultiPV 标注

`annotate_pikafish.py` 读取统一 JSONL，对每个存在人类走法的局面执行 Pikafish MultiPV 搜索，并写入 canonical JSONL。标注阶段不生成 NPY。

```powershell
uv run python scripts\annotate_pikafish.py `
  --input-jsonl data\processed\human_games `
  --output-dir data\processed\pikafish-d10-m5 `
  --depth 10 `
  --multipv 5 `
  --pikafish-threads 1
```

搜索预算用 `--depth N`、`--movetime-ms N` 或 `--nodes N` 三选一。目录输入可用 `--workers N` 并行处理；长任务用 `--resume`，失败游戏默认跳过，`--retry-failed` 才会重试。

每条记录包含 `game_id`、`split`、`ply`、当前 FEN、人类 ICCS 走法和 `teacher`。`teacher.score_kind`/`score` 是 Pikafish 原始当前行棋方分数。每个 MultiPV candidate 包含 rank、首步、原始分数、depth、nodes 和完整 ICCS PV。PV 会回放校验，PV1 必须与 `bestmove` 一致。

`teacher.score` 评估的是走前 FEN 的当前局面，不是人类实战 `move` 走完后的评分。若人类 move 命中 MultiPV candidate，可读取该 candidate 的 score；未进入候选集的人类着没有精确 teacher score。所有 score 的符号都以当前行棋方为正方。

Smoke：

```powershell
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\train-000.jsonl --output-dir data\processed\pikafish-smoke-train-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\validation-000.jsonl --output-dir data\processed\pikafish-smoke-validation-1 --depth 10 --multipv 5
uv run python scripts\annotate_pikafish.py --input-jsonl data\processed\human_games\test-000.jsonl --output-dir data\processed\pikafish-smoke-test-1 --depth 10 --multipv 5
```

确认三次 smoke 零失败、PV 回放和恢复语义正确后，再对全量命令添加 `--resume`。

### 标注吞吐估算

Pikafish 标注的主要瓶颈通常是每个局面的引擎搜索预算，而不是 JSONL 读写或 PGN/XQF 解析。若使用 `--movetime-ms 30`，稳定吞吐常接近每局面 30ms，再叠加少量进程通信和回放校验开销。历史 smoke 中观测过约 `31-32 positions/s`，与 30ms 预算基本一致。

估算总耗时应按局面数而不是棋局数：

```text
hours = positions / (positions_per_second * 3600)
```

多 worker 近似按有效并行度缩短，但会受到 CPU 核数、Pikafish threads、Hash 内存和 IO 调度影响。Pikafish 内置 Hash 在同一进程内可复用部分搜索信息，但不能替代数据级缓存，也不能改变最终训练样本频次。

续跑时注意区分两种速度：

- skip 速度：扫描并跳过已完成游戏，不启动 Pikafish，不能代表真实标注吞吐。
- new rate：新增 positions 的速度，才可用于剩余耗时估算。

## 蒸馏 NPY

`prepare_pikafish.py` 将已完成的 canonical 标注 JSONL 导出为 current-view 蒸馏 NPY。目录输入优先读取完成的 `.jsonl.zst` shard，并跳过 partial 文件。

```powershell
uv run python scripts\prepare_pikafish.py `
  --input-jsonl data\processed\pikafish-d10-m5 `
  --output-dir data\processed\pikafish-distillation `
  --max-candidates 5
```

固定形状字段：

```text
positions               (N, 15, 10, 9)
teacher_cp              (N,)
teacher_score_kinds     (N,)
teacher_scores          (N,)
candidate_action_ids    (N, 5)
candidate_score_kinds   (N, 5)
candidate_scores        (N, 5)
```

合法着使用 ragged 表示：

```text
legal_action_ids      (L,)
legal_action_offsets  (N + 1,)
```

局面 `i` 的合法着为 `legal_action_ids[offsets[i] : offsets[i + 1]]`。动作 ID 为 $90 \cdot from + to$。每个 shard 另有 `games.jsonl`，保存 `game_id` 与样本区间 `[sample_start, sample_end)`。

### Split 与采样

当前 `data/processed/pikafish-distillation/dataset` 采用约 1% validation / 1% test 的固定 split。执行过一次不重导出的 shard 前缀重分配：保留 validation 与 test 各前 11 个 shard，其余按前缀组整体迁移到 train。迁移后规模约为：

```text
train:      1100 shards, 8,987,007 samples
validation:   11 shards,    90,570 samples
test:         11 shards,    90,687 samples
```

这次迁移只改文件归属和 summary，不改变样本内容。后续实验固定使用该 split；test 不参与早停或超参数选择。

训练侧使用 map-style dataset：先建立 shard 累计样本数，再将全局索引映射到 `(shard_id, local_index)`。生产训练使用 block shuffle，而不是 900 万级精确全局 `randperm`：

```text
1. 将全局索引切成固定大小 block
2. 每个 epoch 打乱 block 顺序
3. 每个 block 内局部打乱
4. 再通过 index mapping 读取 shard/local sample
```

默认 `block-size=65536`。该方案在可复现、无重复/无遗漏、吞吐和内存之间折中；精确全局随机只适合小规模 debug 或对照实验。

### Prepare 恢复与进度

当前 `prepare_pikafish.py` 以输出文件完整性作为恢复依据：若某个输入标注 shard 对应的全部 NPY 和 `games.jsonl` 已存在，就直接跳过；缺失或不完整则重新生成该 shard。输出仍位于 `output-dir/dataset/`，汇总写入 `output-dir/dataset_summary.json`。

进度日志统一使用 `[prepare-progress]`，单文件转换过程中约每 5 秒输出一次 records/games/samples；目录转换时输出 `shards=已完成/总数`、当前 shard 前缀和累计样本数。不要为了追求短期 resume 速度跳过样本：重复局面或重复出现的训练样本仍应照常写出，以保持真实样本频次分布。

Pikafish 引擎自身的 Hash/置换表只在同一引擎进程内复用搜索信息，不等同于数据级跨局缓存。若未来实现数据级缓存，应缓存 teacher 结果以减少重复搜索，但仍要按原始出现频次写训练样本。

## Joint Policy/Value 训练

`train_pikafish.py` 读取 ragged 合法着 NPY，训练 `PikafishResNet` 的 8100-logit joint policy head 和 bounded value head。

当前已完成一组可复现的无镜像 20 epoch 训练。它是目前最稳的 c192-b12 Pikafish 蒸馏基线；后续是否发布仍需结合对弈 benchmark 判断。

不传 resume 参数时，脚本自动从 `checkpoint-dir/last.pt` 恢复模型、optimizer、scheduler、AMP scaler 和早停状态。只在完整 epoch 结束时保存 checkpoint。

选模使用 `validation_j_select` 与 early stopping。`min_delta` 会影响 `best.pt` 是否更新：原始指标略有改善但未超过 `min_delta` 时，不覆盖 best checkpoint。比较多个 seed 或多个 epoch 时，应同时看 policy KL、value cp-MAE、sign accuracy 和最终对弈表现，不要只看单个 J 值小数点后微差。

当前基线命令：

```powershell
uv run python scripts\train_pikafish.py `
  --data-dir data\processed\pikafish-distillation\dataset `
  --checkpoint-dir checkpoints\pikafish-c192-b12-lr2e4-vlr2e5-vw1-vs450-w220-nomirror `
  --epochs 20 `
  --learning-rate 2e-4 `
  --value-learning-rate 2e-5 `
  --min-learning-rate 5e-6 `
  --warmup-steps 220 `
  --weight-decay 1e-4 `
  --temperature 100 `
  --value-scale 450 `
  --policy-weight 1 `
  --value-weight 1 `
  --micro-batch-size 2048 `
  --global-batch-size 2048 `
  --max-grad-norm 1 `
  --block-size 65536 `
  --num-workers 8 `
  --prefetch-factor 4 `
  --seed 42
```

该配置不加 `--mirror`，并保持 `micro/global=2048/2048`，即无梯度累积。它恢复了此前约 `39-40` 分钟每 epoch 的训练量级；相比 `micro/global=2048/4096 + mirror`，每个 update 的实际训练计算从约 `8192` 局面降回 `2048` 局面。

### 20 Epoch 结果（2026-09-02）

训练完整跑完 `20` 个 epoch，未早停：

```text
总耗时：47431 秒 = 13 小时 10 分
平均每 epoch：39.5 分钟
epoch 1：42.0 分钟
epoch 20：39.3 分钟
```

最佳指标：

| 指标 | 最佳 epoch | 最佳值 |
|---|---:|---:|
| `validation_j_select` | 20，原始最低 | `0.45931` |
| `joint_loss` | 13 | `2.34185` |
| `policy_loss` | 13 | `2.27617` |
| `value_loss` | 20 | `0.06317` |
| `cp_policy_kl` | 13 | `0.87164` |
| `value_cp_mae_le_300` | 20 | `41.45 cp` |
| `value_cp_mae_all` | 20 | `50.79 cp` |
| `value_cp_mae_gt_300` | 19 | `111.25 cp` |
| `value_sign_accuracy` | 20 | `88.52%` |
| `value_sign_accuracy_gt_300` | 18 | `98.02%` |

实际 `best.pt` 对应 epoch 18，而不是 epoch 20。原因是保存条件要求至少改善 `0.004`：

```text
epoch 18: J_select = 0.459481  -> 保存为 best
epoch 20: J_select = 0.459313  -> 只好 0.000168，未超过 min_delta
```

因此 e20 未覆盖 `best.pt` 是正确行为。e18 和 e20 差异处于 validation 波动量级：e20 的 value 指标略好，e18 的 policy loss / policy KL 略好，建议默认使用 `best.pt`。

从 epoch 1 到 epoch 18/20 的趋势：

```text
J_select:          0.6577 -> 0.4595 / 0.4593   约改善 30%
policy KL:         1.1600 -> 0.8757 / 0.8781   约改善 24%
|cp|<=300 MAE:     61.39  -> 41.49 / 41.45 cp 约改善 32%
all cp MAE:        81.01  -> 50.83 / 50.79 cp 约改善 37%
sign accuracy:     80.53% -> 88.33% / 88.52%
```

### 早期训练健康信号

warmup 在 step `220` 结束，学习率达到峰值：policy `2e-4`，value `2e-5`。step `20 -> 180` 的早期速度约 `0.53s/step`，约 `6800 step/hour`。

`gradient_norm_pre_clip` 虽仍经常超过 `max_grad_norm=1.0`，但趋势下降且 loss 持续改善：

```text
step 20:   46.78
step 220:  18.07
step 800:   1.84
step 1320:  4.90
step 1400:  4.09
```

梯度范数会随 batch 中 mate policy 样本数量波动；持续 `gradient_clipped=true` 本身不是危险信号，需结合原始 norm、loss 和 validation 曲线判断。

### 历史稳定化 Smoke

此前 50-update smoke 使用 `learning-rate=4e-4`、`value-learning-rate=1e-5`、`value-scale=300`、`policy:value=1:0.5`、`micro/global=2048/4096`，结果为 cp-policy KL=`1.7247`、value cp-MAE=`69.80`、$J_{\mathrm{select}}=0.9784`，未出现 value 饱和。它证明了独立 value learning rate 的必要性，但不再是当前首选复现配方。

已知失效：`192x12`、`K=300`、`tau=100`、policy:value=`1:0.5`、`micro/global=2048/4096` 使用 base LR `1.2e-3` 跑完整 epoch 后 value head 饱和；validation value 输出均为 `+1.0`，`tanh` 前激活范围为 `[20.2, 70.6]`。value head 需要独立低学习率或结构调整。

RTX 2060 Super 8GB 参考档位：

| 项目 | 日常吞吐档 | 上限探测 | 回退档 |
|---|---|---|---|
| micro/global batch | `2048 / 2048` | `4096 / 4096` | `1536 / 2048` |
| workers / prefetch | `8 / 4` | `8 / 4` | `8 / 4` |

当前无镜像基线早期稳态约 `0.53s/step`。旧 `2048/4096` 累积配置约 `1.038s/update`，`4096/4096` 上限探测约 `0.820s/update`。Windows 首个 update 包含 DataLoader worker spawn，不能用于稳态吞吐比较。

## 训练目标摘要

首轮训练 Pika 策略和 Pika 价值，不将人类 `move` 混入训练 loss：

1. policy target：纯 `cp` candidate 集构造稀疏软标签；任一 candidate 为 `mate` 时改用 rank 最小 candidate 的硬标签。
2. value target：由 finite `cp` 的 `teacher.score` 构造有界连续标签。
3. human `move`：仅作数据质量与覆盖率统计。

所有 score 保持当前行棋方视角。转换为 current-side-view 时不再额外取反。

Policy 使用 `90 * 90 = 8100` 个联合动作 logits。训练时只对棋规引擎生成的合法着归一化，Top-5 外的合法着不是非法走法，也不应从 softmax 分母移除。

纯 cp policy 使用：

$$
q_i = \frac{\exp((s_i-s_1)/\tau)}{\sum_j \exp((s_j-s_1)/\tau)}
$$

Value 使用：

$$
V(s) = \tanh(\operatorname{clip}(\mathrm{cp}, -900, 900) / K)
$$

当前保留实验矩阵：

```text
value scale K:        300, 400, 600
current K:            450 (当前无镜像基线)
policy temperature:   25, 50, 100 cp
current temperature:  100 cp
policy/value weight:  1:1, 1:0.5, 1:2
current weight:       1:1
mate handling:        policy hard label, excluded from value
human move loss:      disabled
value output:         tanh, bounded to [-1, 1]
```

## cp 与 mate 结论

Pikafish 和 Stockfish 都在搜索内部使用单个可排序整数 `Value`，但在 UCI 输出边界把普通评估和将死评估分成 `score cp` 与 `score mate`。没有官方定义的 cp-to-mate 换算。

因此：

- 训练数据不应把 UCI `mate 3` 写成某个很大的 cp。
- 当前项目让 mate 不进入 value，但进入独立 policy hard-label slice。
- 若未来使用 WDL，应保存实际 Pikafish 二进制输出的 `w d l`，不能套用 Stockfish 国际象棋参数。

建议 score 保持 tagged union：

```text
score_kind: cp | mate
cp:          finite integer | null
mate_moves:  signed integer | null
perspective: side_to_move
```
