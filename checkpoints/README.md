# 训练检查点

每个子目录对应一组训练实验。当前目录包含普通策略、value head、Pikafish 蒸馏以及不同棋局阶段的实验，例如 `resnet-c64-b4-value-start-game`、`resnet-c64-b4-value-end-game`、`resnet-c64-b4-value-whole-game` 和 `pikafish-*`。目录名是实验标识，不单独代表模型质量。

训练输出通常包括：

- `best-epoch-xxxx.pt`：最低 `validation_j_select` 的 checkpoint。
- `best-policy-epoch-xxxx.pt`：最低 `cp_policy_kl` 的 checkpoint。
- `best-value-epoch-xxxx.pt`：最低 `value_cp_mae_le_300` 的 checkpoint。
- `last.pt`：最近一次保存的 checkpoint，可用于中断后恢复。
- `metrics.jsonl`：`train.py` 的逐 epoch/step 训练、验证和测试指标记录。
- `progress.jsonl`：`train_pikafish.py` 的追加事件流，包含启动配置、恢复游标、进度吞吐、训练指标、CUDA 显存和 early-stop 状态；`train_progress` 事件使用平铺训练指标字段。
- `epoch-xxxx.pt`：`train_pikafish.py` 每个完成 epoch 的编号 checkpoint。

checkpoint 保存了模型结构配置以及训练状态。`train.py` 使用 `--resume PATH` 恢复；`train_pikafish.py` 自动从其 `checkpoint-dir/last.pt` 恢复。将选择出的 checkpoint 复制到 `models/` 后，Web 服务才会递归发现并加载它。
