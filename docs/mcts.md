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

重复局面、自然限着和总 ply 必须进入搜索上下文；同一棋盘配不同历史不是同一搜索状态。长将、长捉需要额外保存循环历史。

### P3a 结果

已默认接入模型玩家，不新增 Web 或命令行参数。MCTS 自动接收 `position_counts`、`quiet_plies` 和总 ply，支持理论和棋、三次重复、120 ply 自然限着和 600 ply 最大局长；静着递增，吃子或过河兵重置。长将、长捉精确判罚仍属于后续 P3b。

## P4：转置表和局面缓存

不同走法顺序可能到达相同局面，可以缓存 policy/value 或共享统计。但历史规则存在时，key 不能只使用 `Position`，必须包含影响终局判断的历史摘要。

### P4 结果

尚未实现完整转置表。当前优先保留 P2 root 复用，待 P3b 稳定后再实现带历史上下文的 policy/value 缓存，避免错误合并不同规则状态。

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

## P7：并行搜索

真正的多线程/多进程 MCTS 才需要线程安全统计和 virtual loss。当前 batch selection 已有临时 virtual reservation，单线程路径不需要额外实现 P7。

### P7 结果

暂不实施。当前优先级是 P3b 长将/长捉规则、固定算力下的 `c_puct` 对战实验，以及带历史上下文的 policy/value 缓存。

## 当前深度/广度策略

固定时长仍是正式对弈的主预算，内部根据局面复杂度自动选择 profile：高分支局面偏广度，被将军局面偏深度，普通局面保持平衡。当前 batch=8 的固定算力结果显示根分支覆盖和叶深度波动较大，因此不能仅凭 simulations/s 或单次最大深度判断棋力。

建议用固定 `128/512 simulations`，比较 `exploration=0.75/1.0/1.25/1.5/2.0`，同时记录叶深度、活跃根分支数、访问集中度、固定局面落子和对战胜率。正式对弈保持 `root_temperature=0`；Dirichlet noise 和温度仅在未来自博弈模式中单独开启。

## 总体状态

```text
P0  可测量性        已完成
P1  批量推理        已完成，默认 batch=8
P2  树复用          已完成
P3a 基础历史规则    已完成，默认启用
P3b 长将/长捉       未完成
P4  转置表/缓存     未完成
P5  搜索预算        已完成
P6a prior 校验      已完成
P6b 强制应对 profile 已完成第一版
P6c 动态 c_puct     未完成
P7  并行搜索        暂不实施
```
