# 脚本、数据与训练

所有命令从仓库根目录 `C:\workspace\xiangqi` 执行。项目环境使用 `uv` 管理：

```powershell
python -m pip install uv
python -m uv sync
```

## CUDA 检查

```powershell
uv run python scripts\check_cuda.py
```

脚本会检查 Python、PyTorch、CUDA、显卡名称和显存，并执行一次 GPU 张量计算。应看到 `CUDA available: True` 和 `CUDA verification passed.`。

## 人类棋局数据

### 数据准备约定

输入的人类棋局数据可能来自不同来源。解析阶段统一读取对局元数据和走法，规范化后再转换为训练样本；损坏记录单独写入错误报告，不混入正式数据。

每局从起始 FEN 逐步回放，生成执行当前着法前的局面、起点索引和终点索引。损坏棋局默认整局不导出正式样本，但必须记录来源文件、局号、`ply`、原始 token 和失败原因。数据准备阶段不重新实现中国象棋棋规；合法走法生成属于推理和搜索模块。

统一坐标约定和编码规则由数据编码模块维护，训练数据固定使用 `(15, 10, 9)` 的 `float32` 棋盘张量和起点/终点标签。

正式导出前，先用 `scripts/iccs.sample` 完成解析、编码、标签转换、Dataset 和模型前向的冒烟检查，并人工核对前 10 步。训练循环不解析原始棋谱或坐标字符串。

扫描输入棋谱：

```powershell
uv run python scripts\prepare_data.py --scan-only --output-dir artifacts\scan
```

结构校验：

```powershell
uv run python scripts\prepare_data.py --validate --output-dir artifacts\validate
```

导出策略训练数据：

```powershell
uv run python scripts\prepare_data.py `
  --export `
  --input data\raw\dpxq-99813games.pgns data\raw\WXF-41743games.pgns `
  --output-dir data\processed `
  --shard-size 4096
```

输出为 `positions`、`start_indices`、`end_indices`、终局结果 `values` 的 NPY 分片和 JSONL metadata，默认按 80/10/10 划分 train/validation/test。`--max-games` 可用于冒烟测试。

扫描、校验和导出的中间产物包括扫描统计、成功棋局 JSONL、错误报告、校验摘要、切分清单以及 train/validation/test 数据分片。导出分片包含 `positions`、`start_indices`、`end_indices` 和必要的 metadata；价值标签通过 `--add-values` 单独补充。具体输出目录由命令的 `--output-dir` 决定。

转换旧 NPZ 数据：

```powershell
uv run python scripts\prepare_data.py --convert-npz data\processed\dataset
```

为已有分片补充终局 value 标签：

```powershell
uv run python scripts\prepare_data.py --add-values data\processed\dataset
```

不同来源的人类棋局数据统一整理，并可按 `全局`、`布局`、`残局` 分类输出：

```powershell
uv run python scripts\prepare_chess_book_dataset.py `
  --input-dir data\raw\chess_book-main `
  --output-dir artifacts\chess_book_dataset `
  --formats xqf cbr cbl `
  --shard-size 4096
```

可使用 `--resume` 从解析 checkpoint 继续。解析需要 `cchess`；不支持的输入会记录，不会静默丢弃。数据分类结果用于分段统计和训练集整理；同一来源中的多局数据可展开后统一处理。

## 训练

### 模型与数据契约

模型输入固定为 `(batch_size, 15, 10, 9)` 的 `float32` 张量。通道顺序为红方帅/仕/相/马/车/炮/兵、黑方将/士/象/马/车/炮/卒、当前行棋方；当前行棋方红方为 `1`，黑方为 `0`。

策略模型使用起点和终点两个 90 类输出。模型输出未归一化分数；推理必须先生成当前局面的全部合法起点/终点组合，再对合法组合评分和排序，不能直接拼接两个输出头的最高分。可选 value head 输出当前行棋方视角的标量价值，胜、和、负分别为 `+1`、`0`、`-1`。

训练样本使用当前着法执行前的局面，包含棋盘、起点标签和终点标签；使用 value head 时额外包含价值标签。来源文件、棋局编号和 `ply` 保存在 metadata 中，不随批次传入 GPU。

### 训练原则

- 首轮以监督学习策略基线为目标，不将模型描述为完整象棋引擎。
- 4 GB 显存优先使用 Tiny-ResNet：4 个残差块、64 个通道；先使用 FP32。
- 默认使用普通交叉熵，不预先启用类别权重；只有分段指标显示低频走法明显失真时才做加权或重采样对照。
- 训练集可做左右镜像，并同步转换棋盘列和起点/终点标签；验证集、测试集不做增强，也不做上下翻转。
- 验证集只用于早停和调参，测试集在最终模型确定前保持封存。
- 先完成一个完整 epoch 的性能基准，再估算总训练时间；Windows 下优先从较小的 DataLoader worker 数量开始调试。

### 评估与验收

至少记录起点、终点、完整走法和合法走法 Top-K 指标，并按开局/中局/残局以及红方/黑方分段。最终棋力使用固定时间控制和固定测试集评估，不能只看 validation loss。

数据导出必须验证：标签范围为 `0~89`，棋盘形状正确，ICCS 往返转换一致，连续回放后的行棋方正确，训练/验证/测试棋局无交集，损坏棋局均出现在错误报告中，样本数等于成功棋局有效 `ply` 总数。长棋局不能仅凭样本数量支配整体统计。

### 去重规则

去重发生在数据切分前。规范化起始 FEN、完整 ICCS 着法序列和结果构成内容去重键；来源、棋手等元数据差异不能阻止内容相同的棋局去重。重复来源可保留在 metadata 中，但重复棋局只导出一次样本。

### 来源与限制

不同来源的人类棋局数据和相关解析工具的许可范围不自动相同。发布数据集或模型前必须分别核实来源条款，并保留来源记录。

ResNet 模型：

```powershell
uv run python scripts\train.py `
  --data-dir data\processed\dataset `
  --checkpoint-dir checkpoints\policy `
  --channels 128 --blocks 6 `
  --batch-size 256 --accumulation-steps 1 `
  --learning-rate 0.001 --epochs 100 --patience 5 `
  --num-workers 8 --mirror
```

ResNet 模型加 value head：

```powershell
uv run python scripts\train.py `
  --data-dir data\processed\dataset `
  --checkpoint-dir checkpoints\resnet-value `
  --channels 64 --blocks 4 `
  --batch-size 16 --accumulation-steps 8 `
  --epochs 100 --patience 5 --value-head
```

训练中断后使用 `--resume checkpoints\...\last.pt`。4 GiB 显存不足时降低 batch size 或使用梯度累积。

## QP 残局局面

QP 是残局排局二进制格式。目前可将其解码为残局初始局面和探查结果，但由于没有确认的后续解法着法，不能直接生成策略监督样本。

```powershell
uv run python scripts\reverse_qp.py `
  data\raw\chess_book-main `
  --limit 3725 `
  --output artifacts\qp_probe.json `
  --decoded-output artifacts\qp_decoded_positions.jsonl
```

## 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `check_cuda.py` | 检查 CUDA 和 GPU 张量计算 |
| `prepare_data.py` | PGN 扫描、校验、导出和终局 value 标签 |
| `prepare_chess_book_dataset.py` | 人类棋局 XQF/CBR/CBL 分类导出 |
| `train.py` | ResNet 模型和 value head 训练 |
| `reverse_qp.py` | 解码 QP 残局初始局面 |
| `data_encoding.py` | FEN、ICCS 和局面编码公共底层函数 |

运行测试：

```powershell
uv run pytest
```
