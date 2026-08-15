# 数据准备计划

## 目标

将两组 `.pgns` 棋谱集合离线转换为可直接供 PyTorch `DataLoader` 使用的训练数据：

```text
.pgns 多局棋谱
  -> PGN 风格局解析
  -> FEN + ICCS 着法校验
  -> 逐步生成局面样本
  -> 15 通道棋盘编码
  -> 起点/终点标签编码
  -> 按棋局划分 train/validation/test
  -> 可内存映射的 .npy 数据集
```

训练循环禁止解析 PGN、ICCS 或 FEN 字符串。

## 原始输入

```text
data/raw/dpxq-99813games.pgns
data/raw/WXF-41743games.pgns
```

文件是 PGN 风格的多局文本集合，不按扩展名选择解析器。每局应读取方括号元数据、`FEN`、`Format` 和着法区；只有 `Format` 为 `ICCS` 时才按本计划解析。注意：标签值可能在引号内部跨物理行，例如 `docs/iccs.sample` 中的 `[Red "黑龙江 郭莉萍"]`；解析器不能假设每行都是完整标签，必须支持跨行引号并在规范化时去除标签值中的换行和多余空白，同时保留原始元数据用于追溯。

典型 ICCS 着法为 `C3-C4`：列为 `A-I`，行号为 `0-9`，起点和终点之间通常有连字符。结果标记如 `1-0` 不作为着法。

## 阶段一：原始文件扫描

建议入口：`scripts/prepare_data.py --scan-only`

对每个文件统计：

- 文件大小、局数和来源文件。
- 每局是否存在 `FEN` 和 `Format`。
- `Format` 的取值分布。
- 元数据缺失数量。
- 着法 token 的格式分布。
- 标签跨行、未闭合引号和异常换行数量。
- 结果 `1-0`、`0-1`、`1/2-1/2` 和未知值分布。

此阶段只读原始文件，不生成训练数据。输出 `artifacts/data_scan.json`。

## 阶段二：解析和格式校验

建议入口：`scripts/prepare_data.py --validate`

数据来源已确认是合法对局，因此本阶段不重新实现或验证中国象棋棋规。只检查文件结构、FEN 格式、ICCS 坐标格式、样本字段和棋局边界；合法走法生成器属于后续推理/搜索模块，不是数据导出的前置条件。

已知排除项：DPXQ 的第 7097、7106、7107 局在当前局面回放实现中出现起点无棋子，按用户决定直接跳过。导出报告必须将其计入 `excluded_games=3`，不能混入 `encoding_errors` 或静默丢弃。

每局处理步骤：

1. 解析元数据和 FEN。
2. 先按引号状态合并跨物理行的标签文本，再将规范化后的 FEN 转为 10 行 9 列棋盘。
3. 读取 ICCS 着法，规范化大小写和连字符。
4. 将 ICCS 坐标转换为起点和终点索引 `0~89`。
5. 将 ICCS 走法作为已确认合法的监督标签；生成样本后执行该走法，进入下一局面。
6. 任一步骤失败时记录来源文件、局号、ply、原始 token 和失败原因；默认整局不导出训练样本，不能把无法解析的截断局混入正式数据。保留损坏棋局的有效前缀仅用于诊断。

必须使用同一套坐标函数完成：

```python
iccs_to_indices(move)
indices_to_iccs(start_index, end_index)
encode_fen(fen)
apply_move(position, start_index, end_index)
```

训练前门禁：先用 `docs/iccs.sample` 完成小样本端到端流程 `解析 -> 编码 -> 标签转换 -> NPY -> Dataset -> 模型前向`；该流程未通过前，不得开始全量导出或训练。无需等待完整棋规验证。

坐标约定固定为：ICCS 列 `A-I` 映射到 `0-8`，ICCS 行 `0-9` 映射到棋盘行 `0-9`；FEN 从黑方一侧开始，因此 `fen_row = 9 - iccs_row`。棋盘张量统一使用 ICCS 行顺序，不能在不同模块重复翻转。用 `docs/iccs.sample` 的前 10 步进行人工核对，并验证包含跨行 `[Red]` 标签的整局可以正常解析；执行整局后还要核对棋盘、轮次和结果状态一致。未闭合引号或无法判定标签边界时，该局进入错误报告，不得静默拼接。

输出：

```text
artifacts/validated_games.jsonl
artifacts/data_errors.jsonl
artifacts/validation_summary.json
```

每条成功棋局至少保留 `game_id`、来源文件、局号、规范化元数据、起始 FEN、完整 ICCS 着法序列、最终结果和输入文件哈希。

棋局状态：解析并保留 FEN 中的当前行棋方、半回合计数和回合号，用于正确生成连续局面；重复局面、长将/长捉等规则不在数据准备阶段重新判定。该数据集的目标是合法棋谱监督学习，不是棋规验证器。

## 阶段三：去重和按棋局划分

建议入口：`scripts/split_dataset.py`

1. 用来源、规范化起始 FEN 和完整着法序列生成稳定的 `game_id`。
2. 两个来源合并前去重，保留重复来源列表；去重必须发生在切分之前。
3. 切分单位是棋局，不能按单步样本或局面随机划分；同一局的所有 ply 必须进入同一集合。
4. 推荐比例为 `80% train / 10% validation / 10% test`。训练集用于拟合，验证集只用于早停和调参，测试集在最终模型确定前不得查看指标。
5. 如果当前阶段只保留训练集和验证集，则使用 `90%/10%`，但必须额外生成一份封存的最终评估清单，不能反复用验证集报告最终成绩。
6. 优先按 `Event`、比赛日期、棋手组合或来源批次做分组；同组棋局不得跨集合。缺少可靠分组字段时，至少按 `game_id` 随机分配，并对起始 FEN、完整着法序列和元数据做近重复检查。
7. 在满足分组约束的前提下，尽量让各集合的来源、结果、红黑方和棋局长度分布接近；这是分层目标，不得为了分层拆开同一分组。
8. 固定随机种子、切分算法版本和输入文件哈希，并保存划分清单，保证后续实验可复现。

输出：

```text
artifacts/train_games.jsonl
artifacts/validation_games.jsonl
artifacts/test_games.jsonl
artifacts/split_manifest.json
```

## 阶段四：生成训练样本

建议入口：`scripts/prepare_data.py --export`

每个 ply 生成一条样本，样本使用执行当前着法之前的局面：

```text
(position_before_move, start_index, end_index)
```

棋盘张量固定为：

```text
shape: (15, 10, 9)
dtype: float32
```

通道定义：

```text
0~6:  红方帅/仕/相/马/车/炮/兵
7~13: 黑方将/士/象/马/车/炮/卒
14:   当前行棋方（红方为1，黑方为0）
```

起点和终点标签均为 `int64`，范围 `0~89`。每条样本还要保留 `game_id`、`ply`、当前行棋方、结果和阶段索引，至少在旁路元数据中可追溯。输出使用三个未压缩 `.npy` 数组组成一个分片，避免一次性构造过大的单个数组，并允许训练进程使用内存映射读取；分片只允许在棋局结束后刷新，即使超过 `shard_size` 也不能把同一 `game_id` 拆到多个分片：

```text
artifacts/dataset/train-000-positions.npy
artifacts/dataset/train-000-start_indices.npy
artifacts/dataset/train-000-end_indices.npy
```

每组三个数组包含：

```text
positions: float32  [N, 15, 10, 9]
start_indices: int64 [N]
end_indices: int64 [N]
```

元数据另存为 JSONL 或 parquet，包含 `game_id`、`source_file`、`ply`、`side_to_move`、`result`、`phase`、编码版本和数据分片位置。

导出前记录数据版本、脚本版本、输入文件 SHA-256、随机种子、棋盘通道顺序、ICCS 坐标约定和切分算法版本；这些信息写入 `dataset_summary.json` 和 `split_manifest.json`。

## 阶段五：数据增强

镜像不放在原始棋谱解析阶段执行。阶段四先导出未增强的规范数据并完成基线训练；确认坐标和标签无误后，在训练 Dataset 读取训练样本时执行确定性的左右镜像。这样不会污染验证/测试分布，也避免把训练数据和缓存体积无条件翻倍。

如果实测 CPU 镜像变换成为瓶颈，可在训练集上单独生成镜像缓存分片，但不得镜像验证集或测试集；缓存必须记录来源 `game_id`、原样本索引和增强版本，避免重复增强。

训练阶段的左右镜像规则：

- 只交换棋盘列 `A-I` 的左右位置。
- 同步转换起点和终点列。
- 不做上下翻转。
- 当前行棋方和结果标签不变。
- 验证集和测试集不做随机增强，测试集还必须保持完全封存。

## 数据验收标准

导出前必须通过：

- 所有 `positions` 形状为 `[N, 15, 10, 9]`。
- 所有起点、终点标签都在 `0~89`。
- 每条标签走法都能通过 ICCS 格式解析，并能完成连续局面更新。
- `indices_to_iccs(iccs_to_indices(move))` 与规范化 ICCS 一致。
- 训练集和验证集的 `game_id` 无交集。
- 若生成测试集，训练集、验证集和测试集的 `game_id` 两两无交集。
- 若使用事件、日期或批次分组，任何分组键不得跨集合。
- 验证集和测试集的来源、结果、红黑方和棋局长度分布不得出现明显缺失。
- 训练/验证/测试统计既要按样本报告，也要按棋局报告；不能让长棋局凭样本数量支配整体指标。
- 起点、终点和完整走法的频次分布必须导出，检查极端稀有标签、空类别和训练/验证分布漂移。
- `side_to_move` 与棋盘通道编码一致，首个样本和每次执行走法后的行棋方交替正确。
- 对 `docs/iccs.sample`、初始局面、吃子和局面连续更新建立固定回归样例；不把棋规合法性回归测试作为数据导出的前置条件。
- 数据文件和元数据可通过哈希与样本计数互相校验，分片缺失或顺序错乱时验证脚本必须失败。
- 样本来源和棋谱使用许可已记录；若来源限制再分发，模型和数据发布计划必须单独处理。
- 损坏棋局均出现在错误报告中，没有静默丢失。
- 样本数量等于成功棋局所有有效 ply 的总和。
- 随机抽样的局面、标签和执行后局面可人工复核。

建议入口：`scripts/validate_dataset.py`，输出 `artifacts/dataset_summary.json`。只有验收通过后才开始训练。

## 交付物

```text
artifacts/data_scan.json
artifacts/validated_games.jsonl
artifacts/data_errors.jsonl
artifacts/validation_summary.json
artifacts/split_manifest.json
artifacts/dataset/train-*-positions.npy
artifacts/dataset/train-*-start_indices.npy
artifacts/dataset/train-*-end_indices.npy
artifacts/dataset_summary.json
```

详细模型输入输出约定见 [模型接口设计](model_interface.md)，总体训练取舍见 [训练计划](plan.md) 和 [象棋训练实践调研](training_best_practices.md)。
