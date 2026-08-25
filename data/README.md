# 数据目录

数据处理流程从 `data/raw/` 的原始棋谱开始，经过统一 JSONL，再按训练配置导出 NPY。大文件和生成数据不应直接混入代码目录之外的临时路径，以便复现实验。

## 目录结构

```text
data/raw/
	PGN/XQF 棋谱和 xqp 数据

data/processed/human_games/
	train-*.jsonl
	validation-*.jsonl
	test-*.jsonl
	duplicates.jsonl
	dataset_summary.json

data/processed/<run>/
	标注或转换配置、数据摘要、处理进度和 JSONL/N​​PY shard
```

`human_games` 是统一后的逐局数据集；split 按棋局 ID 划分，避免同一棋局跨 train、validation 和 test。`duplicates.jsonl` 保留被去重记录的审计信息。Pikafish 标注运行还会写入 `annotation_config.json`、失败事件流和 `processed_game_ids.txt`，支持安全续跑。

`prepare_data.py` 输出普通策略训练使用的 NPY shard；`prepare_current_view.py` 输出 `current_view` 或 `current_view_15` 等 current-side-view 数据集。后者会把棋盘和走法转换为当前行棋方视角，并同时保存 value 标签，因此不应与普通人类棋局 JSONL 混用。
