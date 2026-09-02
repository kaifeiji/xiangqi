# 实验结论台账

本文记录从历史会话和当前仓库产物中筛出实验结论。详细设计分别见 [mcts.md](mcts.md)、[pikafish.md](pikafish.md)、[data-training.md](data-training.md) 和 [architecture.md](architecture.md)。

证据强度约定：

- `strong`：有完整日志、benchmark JSON 或 checkpoint 指标支撑。
- `medium`：有短实验或局部复盘支撑，样本量有限。
- `weak`：历史经验或小样本观察，只作为避免重复踩坑的提示。

## 当前推荐基线

| 领域 | 当前基线 | 证据 |
|---|---|---|
| Pikafish 蒸馏训练 | `c192-b12`、无镜像、`micro/global=2048/2048`、policy LR `2e-4`、value LR `2e-5`、`value-scale=450`、`temperature=100` | `strong` |
| 模型对局 benchmark | 主流开局库成对换色，盘数为 `MAINSTREAM_OPENINGS.len() * 2` | `strong` |
| MCTS 固定算力 | 以 `mcts_simulations` 控制：`0 / 1000 / 5000 / 10000` | `strong` |
| 反循环规则 | 模型方优先选择不回到历史局面的合法着，三次重复作和棋兜底 | `strong` |
| Web 推理模型 | 训练 checkpoint 先导出 FP32 ONNX；Rust 不直接加载 `.pt` | `strong` |

## Pikafish 蒸馏训练

### 无镜像 20 epoch 基线（strong）

命令见 [pikafish.md](pikafish.md) 的“当前基线命令”。关键配置：

```text
checkpoint-dir:      checkpoints\pikafish-c192-b12-lr2e4-vlr2e5-vw1-vs450-w220-nomirror
micro/global batch:  2048 / 2048
learning rate:       2e-4
value learning rate: 2e-5
min learning rate:   5e-6
warmup steps:        220
value scale:         450
policy/value weight: 1 / 1
max grad norm:       1
block size:          65536
mirror:              disabled
```

20 epoch 完整跑完，未早停：

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

`best.pt` 对应 epoch 18。epoch 20 的 `J_select` 只比 epoch 18 好 `0.000168`，未超过 `min_delta=0.004`，因此没有覆盖 best checkpoint。默认使用 `best.pt`，不要手动追逐 e20 的微小原始优势。

### Value head 饱和与学习率（strong）

`base LR=1.2e-3` 曾导致 value head 在完整 epoch 后饱和：validation value 输出几乎全为 `+1.0`，`tanh` 前激活范围约 `[20.2, 70.6]`。单步 value-only 学习率扫描显示：

| value LR | 更新后 value 均值 | 判定 |
|---:|---:|---|
| `1e-5` | `0.022` | 稳定 |
| `4e-5` | `0.217` | 未饱和 |
| `1e-4` | `0.470` | 明显漂移 |
| `4e-4` | `0.957` | 接近饱和 |
| `1.2e-3` | 约 `1.000` | 单步完全饱和 |

结论：Pikafish joint policy/value 训练必须给 value head 独立低学习率。当前基线使用 policy `2e-4`、value `2e-5`。

### Mirror 取舍（medium）

镜像增强理论上合理，但在当前训练轨道下实际成本接近翻倍。此前 `micro/global=2048/4096 + mirror` 相当于每个 update 处理约 `8192` 个局面，明显慢于无镜像 `2048/2048` 的约 `2048` 局面/update。当前基线暂不启用 mirror，先用非镜像配方稳定训练和对局验证。

### 阶段训练经验（weak）

历史上分别用残局、布局、全局训练后，都出现“3 epoch 很快收敛，但实际对弈变傻”的观察。该结论缺少当前文档化指标和可复查 benchmark，不作为正式配方依据，但保留为警示：不要只看训练/validation 快速收敛，应以固定开局库换色 benchmark 和关键局面复盘验证棋力。

## 数据与采样

### 1% validation/test split（strong）

`data/processed/pikafish-distillation/dataset` 已迁移到约 1% validation / 1% test，不重导出样本，只调整 shard 前缀归属。当前规模：

```text
train:      1100 shards, 8,987,007 samples
validation:   11 shards,    90,570 samples
test:         11 shards,    90,687 samples
```

结论：后续实验固定该 split；test 不参与早停、调参或 checkpoint 选择。

### Block shuffle（strong）

900 万级样本不默认做精确全局 `randperm`。当前采用 map-style dataset + index mapping + block shuffle：块间随机、块内随机，再映射到 `(shard_id, local_index)` 读取样本。默认 `block-size=65536`。

结论：该策略在可复现、无重复/无遗漏、吞吐和内存之间折中。精确全局随机只用于小规模 debug 或对照实验。

### 合法着 mask 分工（medium）

普通人类棋局训练中，监督标签本身应合法，因此训练 loss 不需要每 batch 生成完整合法着 mask。validation/test 的完整着法指标仍可衡量真实落子质量。Pikafish 蒸馏不同：teacher candidate 需要在全合法着集合上归一化，因此 `train_pikafish.py` 必须读取 ragged `legal_action_ids`。

## Pikafish 标注

### teacher score 语义（strong）

`teacher.score` 是走前 FEN 的当前行棋方评估，不是人类实战 move 走完后的评分。若人类 move 命中 MultiPV candidate，可读取对应 candidate score；未进入候选集的人类着没有精确 teacher score。

结论：训练 value target 只能从 root `teacher.score` 构造；人类 move 只用于覆盖率和质量审计，不混入当前 loss。

### 标注吞吐估算（medium）

Pikafish 标注主要受每局面的引擎搜索预算限制。历史 `--movetime-ms 30` smoke 观测约 `31-32 positions/s`，与 30ms 预算吻合。估算耗时按 positions，不按 games：

```text
hours = positions / (positions_per_second * 3600)
```

skip 速度不代表真实吞吐；只有新增 positions 的 `new_rate` 可用于剩余耗时估算。

## MCTS 与推理

### 真实 simulation 计数（strong）

`batch=32` 下曾出现同一未展开叶在一个 batch 内被重复选择并多次回传，导致 `1000 sims` 看起来更快但语义不正确。修复后重复叶选择会释放 virtual loss，不计入有效 simulation。

典型正确日志：

```text
simulations=1000
batch=32
inference_batches=33
evaluated_leaves=1000
average_onnx_batch=30.30
duplicate_leaf_selections=0
```

结论：修复后 `1000 simulations` 表示 `1000` 个真实 NN 叶评估；旧 `3s/1000 sims` 不能作为等价性能基准。

### c64 与 c192 延迟差异（strong）

已观测：

```text
c64-b4-current-mirror.onnx / 1000 sims / batch=32
total=2.458s
onnx_primary=0.212s

c192-b12 / 1000 sims / batch=32
total=5.932s
onnx_primary=4.403s
selection=0.003s
CPU 扩展等约 1.529s
```

结论：c192 的主要瓶颈是 ONNX 端到端评估，其次是规则/节点扩展；selection 本身不是瓶颈。不同模型大小不能横向比较“每次推理 2ms”这类数字。

### FP16 / TensorRT 取舍（medium）

FP16 实验已移除：当前生产路径保留 FP32 ONNX + CUDA EP，避免可见数值误差影响棋力判断。TensorRT 仍可作为未来性能实验，但不能消除 CPU 扩展成本。

以 c192 基线估算：若 TensorRT FP16 让 ONNX 部分 `2x`，总时延约从 `5.9s` 降到 `3.8s`；要稳定进入 `3s`，还需同时优化 CPU 扩展。

### Q Guard 与 Policy Temperature（medium）

历史对局扫描显示，MCTS 不是普遍 selected move 异常，`selected_move` 都能在 root children 中找到，且通常等于 visits 第一。但有 24 个 root 满足：Q 最佳候选访问数至少 `25`，且当前方 Q 比 visits 第一高至少 `0.15`。这类点集中在 tactical/送子历史存档和当前 1000 sims benchmark。

当前采用保守 Q guard 默认值：

```text
MCTS_Q_GUARD_MIN_VISITS=25
MCTS_Q_GUARD_MIN_GAP=0.15
```

固定 FEN 测试结论：

| 局面 | sims | exploration | policy temp | 结果 |
|---|---:|---:|---:|---|
| game6 | 1000 | 1.25 | 1.25 | 选 `F2-E2`，Q 最佳 `F2-H2` |
| game6 | 1000 | 0.75 | 2.0 | 仍选 `F2-E2`，但 `F2-G2` visits 接近 |
| game6 | 5000 | 0.75 | 1.25 | 选 `F2-G2`，和 Q 最佳一致 |
| game7 | 1000 | 1.25 | 1.25 | `C7-E6` 被 `prior=0.625` 锁住 |
| game7 | 1000 | 0.75 | 1.25/1.5 | 改选 `I3-I4`，和 Q 最佳一致 |
| game7 | 1000 | 0.75 | 2.0 | 又回到 `C7-E6`，过度软化变差 |

结论：policy 不是整体过尖，合法着数不少于 8 的 root 中 top1 prior 均值约 `0.226`。问题是少数局面存在错误高 prior 或 visits/Q 未收敛。训练侧暂不建议单纯提高 distillation temperature；推理侧保留 Q guard。`MCTS_EXPLORATION=0.75` 只在 game7 探针中优于 `1.25`，尚不足以替换默认值；`MCTS_POLICY_TEMPERATURE=1.5` 未显示明确额外收益，`2.0` 在 game7 退化。后续 benchmark 应固定 temperature=`1.25`，只比较 exploration=`0.75/1.25`。

## Benchmark 结果

### 纯 policy 对局的循环问题（strong）

早期 `mcts_simulations=0` benchmark 中，大量结果是长将、长捉或三次重复。这说明纯 deterministic policy 容易进入循环线，尤其两个相近模型互弈时更明显。

现有 benchmark JSON 摘要：

| 文件 | MCTS | 完成 | 结果分布摘要 |
|---|---:|---:|---|
| `20260902-122802-702.json` | 0 | 22/22 | `draw_repetition=8`，长将/长捉合计 8，普通胜负 6 |
| `20260902-123803-006.json` | 0 | 22/22 | `draw_repetition=10`，长将/长捉合计 5，普通胜负 7 |
| `20260902-124318-417.json` | 0 | 22/22 | 长将合计 9，普通胜负 12，`draw_natural_limit=1` |
| `20260902-125024-722.json` | 0 | 26/26 | 长将/长捉合计 12，普通胜负 13，`draw_natural_limit=1` |
| `20260902-131140-312.json` | 0 | 26/26 | 长将/长捉合计 14，普通胜负 11，`draw_natural_limit=1` |
| `20260902-132620-231.json` | 0 | 26/26 | 普通胜负 23，`draw_natural_limit=3`，无长将/长捉/三次重复 |

结论：简化反循环规则显著改善纯 policy benchmark 的循环终局分布；最终判断还需要继续跑同模型、同开局库的对照。

### Benchmark 暂停与可恢复存档（strong）

benchmark 不再将取消视为终止结果。暂停时已完成对局和中断盘的 snapshots 都会持久化；恢复时从该盘的 `initial_fen` 和最后一个 snapshot 重建 `Game`，继续未完成对局。存档不再记录 `moves` 或 `final_fen`，避免并行维护两份可能不一致的对局状态。

恢复只适用于未完成且未失败的任务。任务 JSON 在服务重启后从 `BENCHMARK_PATH` 重新加载，因此暂停任务不依赖原进程存活。

### 1000 sims benchmark 成本（medium）

`20260902-132851-469.json` 使用 `mcts_simulations=1000`，当前只完成 2/26 盘，结果为红胜 1、黑胜 1。两盘耗时分别约 `430.6s` 和 `566.8s`。

结论：c192 + 1000 sims 的完整 26 盘 benchmark 成本很高，不适合频繁调参时作为第一层筛选。建议先用纯 policy 或少量固定局面复盘做快速筛，再跑完整 MCTS benchmark。

## 已废弃或暂缓方向

### Alpha-beta + ResNet value（strong）

运行时 alpha-beta + ResNet value 方向已撤掉。原因是传统 alpha-beta 需要极高节点吞吐，而 ResNet value 每节点推理成本过高。当前 Web 对弈主路径使用 policy/value + MCTS。

### Python MCTS + Rust 棋规逐节点 FFI（strong）

逐节点 FFI 会把局面序列化/解析成本放大到 MCTS 热路径。当前方向是把棋规、规则状态和搜索整体留在 Rust 内部，Python 只负责离线数据与训练。

### FP16 默认部署（medium）

出于棋力稳定性和可解释性，当前不默认部署 FP16 ONNX。未来若重启该实验，必须以固定战术局面、policy KL/value MAE 和同模型对战一起验收。
