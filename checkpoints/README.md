# 训练检查点

每个子目录对应一组训练实验。当前目录包含普通策略、value head 以及不同数据视角/棋局阶段的实验，例如 `resnet-c64-b4-value-start-game`、`resnet-c64-b4-value-end-game` 和 `resnet-c64-b4-value-whole-game`。目录名是实验标识，不单独代表模型质量；具体指标以该目录的 `metrics.jsonl` 为准。

训练输出通常包括：

- `best.pt`：validation 指标最佳的 checkpoint。
- `last.pt`：最近一次保存的 checkpoint，可用于中断后恢复。
- `metrics.jsonl`：逐 epoch/step 的训练、验证和测试指标记录。

checkpoint 保存了模型结构配置以及训练状态。可使用 `scripts/train.py --resume PATH` 继续训练；将 checkpoint 复制到 `models/` 后，Web 服务才会递归发现并加载它。
