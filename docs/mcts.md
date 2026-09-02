# MCTS 优化记录

本文记录当前中国象棋 MCTS 的优化提案、实施结果和后续优先级。正式对弈使用固定时长作为硬预算；固定 simulation 数只用于可重复 benchmark。

## P0：可测量性

固定以下指标：每秒 simulation 数、平均/最大叶深度、根节点 visits/Q/prior、固定局面落子和对战结果。

### P0 结果

已新增 MCTS benchmark、搜索统计和回归测试。模型 `models/resnet-c64-b4-current-mirror.pt` 在 CPU、1 秒预算下为 52、51、55 simulations，平均约 52.24 simulations/s，平均叶深度约 4.69 ply，最大深度 11 ply，三次均选择 `B2-E2`。结果保存于 `benchmark/mcts-baseline.json`。

## P1：批量推理

批量收集叶节点，一次完成 policy/value 推理，再分别 backup，以提高 GPU 利用率和单位时间 simulation 数。

### P1 结果

已实现 batch 叶节点评估，默认 `batch_size=8`，使用临时 virtual reservation 分散同批 selection。CPU 1 秒吞吐从约 52.24 提升到约 77.29 simulations/s；RTX 2060 SUPER 已验证 CUDA 可用。默认设备为 CUDA，无 CUDA 时回退 CPU。

## P2：树复用

保留上一轮搜索的选中子树，在下一回合继续使用已有 visits、value_sum 和展开状态。

### P2 结果

已完成 root 子树复用。MCTS 可复用旧 root、直接子节点或两层后代；两个模型玩家会跨回合保存 MCTS 实例。找不到对应局面时自动创建新 root，单合法着局面仍直接返回。

## P3：历史规则

重复局面、自然限着和总 ply 必须进入搜索上下文；同一棋盘配不同历史不是同一搜索状态。当前实现不做复杂长将/长捉归责，而是用模型选着阶段的反循环过滤降低死循环概率。

### P3a 结果

已默认接入模型玩家，不新增 Web 或命令行参数。MCTS 自动接收规则历史，支持理论和棋、三次重复、120 ply 自然限着和 600 ply 最大局长。

模型方使用简化反循环策略：若某局面存在不会回到历史局面的合法着法，则 MCTS/policy/Pikafish 只在非重复着法中选择；只有全部合法着法都会重复时才允许重复。三次重复统一作为和棋终局兜底。

## P4：转置表和局面缓存

不同走法顺序可能到达相同局面，可以缓存 policy/value 或共享统计。但历史规则存在时，key 不能只使用 `Position`，必须包含影响终局判断的历史摘要。

### P4 结果

尚未实现完整转置表。当前优先保留 P2 root 复用；若实现 policy/value 缓存，key 必须包含足够历史摘要，避免错误合并不同规则状态。

## P5：搜索预算

增加固定 simulation 数和最大搜索深度，使 benchmark 可复现，并防止 batch 最后一批超过预算或异常线路递归过深。

### P5 结果

已支持 `max_simulations` 和 `max_depth`。固定算力使用 `--mcts-time 0 --max-simulations N`，batch 最后一批严格截断；默认最大深度为 `256`。RTX 2060 SUPER 上固定 128 simulations、batch=8 的三次结果均严格为 128，平均叶深度为 1.75、1.88、2.38 ply，最大深度为 3、4、4 ply，活跃根分支为 7、14、14。

## P6：PUCT 和先验处理

候选方向包括 prior 校验、强制应对深度补偿、动态 `c_puct`、根节点噪声和自博弈温度。正式对弈不使用 Dirichlet noise 或温度采样；不做未经验证的 Top-p 硬剪枝。

### P6 结果

P6a 已完成 prior/value 防御性校验：prior 必须恰好覆盖合法着、有限且非负，value 必须为有限的 `[-1, 1]` 数值。50 个随机局面压力测试结果为 `prior_failures=0`、`illegal_moves=0`、`exceptions=0`。

P6b 已完成第一版自动 profile：根节点被将军时使用 `exploration=0.9`、`max_depth=384`；合法着数不少于 35 时使用 `exploration=1.5`；其他局面保持 `exploration=1.25`、`max_depth=256`。不删除合法应将。

P6c 目前只有按根节点复杂度选择的静态 profile，尚未加入随 visits 动态变化的 `c_puct` 公式。当前代码也没有“policy 概率 25 次方锐化”的实现证据，因此不根据该假设强行偏向深搜。

根节点最终选着以 visits 最大为主；Q guard 只在 Q 最佳候选的 visits 不低于 `MCTS_Q_GUARD_MIN_VISITS`，且当前方 Q 比 visits 第一高出 `MCTS_Q_GUARD_MIN_GAP` 时介入。默认值为 `25`、`0.15`，可通过环境变量调整。

## P7：并行搜索

真正的多线程/多进程 MCTS 才需要线程安全统计和 virtual loss。当前 batch selection 已有临时 virtual reservation，单线程路径不需要额外实现 P7。

### P7 结果

暂不实施。当前优先级是固定算力下的 `c_puct` 对战实验，以及带历史上下文的 policy/value 缓存。

## 当前深度/广度策略

固定时长仍是正式对弈的主预算，内部根据局面复杂度自动选择 profile：高分支局面偏广度，被将军局面偏深度，普通局面保持平衡。当前 batch=8 的固定算力结果显示根分支覆盖和叶深度波动较大，因此不能仅凭 simulations/s 或单次最大深度判断棋力。

建议用固定 `128/512 simulations`，比较 `exploration=0.75/1.0/1.25/1.5/2.0`，同时记录叶深度、活跃根分支数、访问集中度、固定局面落子和对战胜率。正式对弈保持 `root_temperature=0`；Dirichlet noise 和温度仅在未来自博弈模式中单独开启。

## 近期实测记录（2026-09-02）

### 批量选择与真实 simulation 计数

`batch=32` 时已修复同一未展开叶在一个 batch 内被重复选择并多次回传的问题。重复叶选择会释放 virtual loss，不再计入有效 simulation。修复后 `1000 simulations` 表示 `1000` 个真实 NN 叶评估；此前约 `3s/1000 sims` 的结果包含重复叶回传，不能作为等价基准。

典型日志口径：

```text
simulations=1000
batch=32
inference_batches=33
evaluated_leaves=1000
average_onnx_batch=30.30
duplicate_leaf_selections=0
```

这说明 batch 已基本满载，selection 也不是主要瓶颈。

### c64 与 c192 推理成本差异

不同模型大小不能横向比较延迟。已观测：

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

`onnx_primary` 是端到端评估时间，包含输入编码、ORT tensor、CPU/GPU 传输、`session.run()`、输出读取和合法着 softmax，不等于纯 GPU kernel 时间。对上述 c192 记录，平均每个 ONNX batch 约 `133ms`，每个叶局面端到端约 `4.4ms`。

### 深度解释

`average_leaf_depth=3.5`、`max_leaf_depth=5` 表示大多数 NN value 评估发生在根后约 3 到 4 个半回合，是偏浅的 policy-guided MCTS。后期出现 `max_leaf_depth=16` 是健康信号：说明访问逐渐集中后，主变能继续向下扩展；但单个最大值不能代表整体搜索深度。

后续分析应记录 depth 分位数，而不只看平均和最大：`P50/P90/P99/max` 比 `average/max` 更能区分“整体浅”与“少数主变深入”。

### 根节点 Q Guard

最终落子仍以 visits 最大为主，这是 AlphaZero 风格的标准选着方式。但历史对局显示，少数局面中 policy prior 会把 visits 锁在当前方 Q 明显较差的候选上。当前加入保守 Q guard：

```text
MCTS_Q_GUARD_MIN_VISITS=25
MCTS_Q_GUARD_MIN_GAP=0.15
```

若 Q 最佳候选访问数达到下限，且当前方 Q 比 visits 第一高出阈值，则最终选择 Q 最佳候选。扫描历史 `benchmark/` 和 `xiangqi-test/` 的 1255 个 MCTS root，该默认值触发约 24 次；不是把 MCTS 改为纯 Q 贪心，而是拦截明显 visits/prior 失真。

固定 FEN 探针结果：

```text
game6, 1000 sims, exploration=1.25: F2-E2，Q 最佳 F2-H2
game6, 5000 sims, exploration=0.75: F2-G2，和 Q 最佳一致

game7, 1000 sims, exploration=1.25: C7-E6，被 prior=0.625 锁住
game7, 5000 sims, exploration=1.25: 改为 E8-G8，但仍未达到最佳 Q
```

该结果说明：一部分问题可通过更多 simulations 或更低 exploration 缓解；一部分是局部高 prior 错误，需要 Q guard 兜底或后续训练诊断。

### Policy Temperature 探针

推理温度 `MCTS_POLICY_TEMPERATURE` 会把 logits 转 prior 时软化或锐化分布。历史 benchmark root 统计显示整体 policy 并不普遍过尖：合法着数不少于 8 的 root 中，top1 prior 均值约 `0.226`，P90 约 `0.349`，top1 大于 `0.5` 的比例约 `57/1220`。

固定 FEN 探针结果显示，温度不是单调收益：

```text
game6, 1000 sims, exploration=0.75:
temp=1.25/1.5/2.0 均仍选 F2-E2，但 temp=2.0 使 F2-G2 visits 接近。

game7, 1000 sims, exploration=0.75:
temp=1.25/1.5 均选 I3-I4，和 Q 最佳一致；
temp=2.0 又回到 C7-E6，说明过度软化会让搜索变散。
```

因此当前不把 `MCTS_POLICY_TEMPERATURE` 默认提高到 `2.0`。若实验，可优先尝试 `1.5`，并和 `MCTS_EXPLORATION=0.75` 成对测试。

### FP16 与 TensorRT 取舍

当前生产路径保留 FP32 ONNX + CUDA EP，CUDA 不可用时回退 CPU。FP16 实验已移除：不保留 FP16 ONNX、`MCTS_ONNX_PRECISION` 或 `half` 依赖，避免可见数值误差影响棋力判断。

TensorRT 仍是可能的性能方向，但不是当前默认依赖。以 c192 基线估算：

```text
总耗时        5.895s
ONNX CUDA EP  4.163s
其余 CPU      1.732s
```

TensorRT 只能压缩 ONNX 这部分；若 TensorRT FP16 获得 `2x` ONNX 加速，总时延约 `3.8s`。要进入 `3s`，需要 ONNX 约 `3.3x` 加速，或同时把 CPU 扩展从约 `1.7s` 优化到约 `1s`。

### 仍值得优化的方向

1. `RuleState::child()` 与候选展开的历史状态复制。历史越深，复制越贵；可考虑父指针或共享不可变历史。
2. 输入 buffer 与输出解析复用，减少每个 batch 的分配和 logits 后处理成本。
3. 带历史摘要的 policy/value 缓存或转置表，但不能只用棋盘 key。
4. 固定算力下的 `c_puct` 对战实验；不要无验证地加入 Top-p、Early Stopping 或多线程队列。

## 总体状态

```text
P0  可测量性        已完成
P1  批量推理        已完成，默认 batch=8
P2  树复用          已完成
P3  基础历史规则/反循环 已完成，默认启用
P4  转置表/缓存     未完成
P5  搜索预算        已完成
P6a prior 校验      已完成
P6b 强制应对 profile 已完成第一版
P6c 动态 c_puct     未完成
P7  并行搜索        暂不实施
```
