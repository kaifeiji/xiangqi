# Benchmark Results

后台 benchmark 任务会在此目录按本地开始日期时间保存 JSON 结果，例如 `20260902-123456-789.json`。可通过 `BENCHMARK_PATH` 修改输出目录。

## 赛制

Web benchmark 比较两个模型 A/B。盘数由主流开局库决定：每个开局走一组 A 执红、B 执黑，再交换红黑走一盘，因此总盘数为 `MAINSTREAM_OPENINGS.len() * 2`。当前界面不让用户手动指定盘数，避免少量随机开局导致结论不稳定。

每盘开局记录为 ICCS + 中文记谱，例如：

```text
B2-E2（炮二平五）
```

完全禁用开局库会让策略模型高度集中到少数首着，例如当头炮，样本多样性更差；共享同一开局库并交换红黑更适合做模型对照。

## 保存内容

单个 benchmark JSON 保存：模型 ID、MCTS simulation 数、请求盘数、起止时间、取消/失败状态和每盘摘要。

每盘保存：

- `number`：盘号。
- `opening_move`：开局首着和中文记谱。
- `started_at_ms` / `finished_at_ms` / `elapsed_ms`：单盘计时。
- `result`：胜负或和棋原因。
- `total_plies` / `rule60`：总 ply 和自然限着计数。
- `initial_fen`：该盘初始 FEN。
- `snapshots`：按存档格式保存的逐步局面，供观看和复盘。
- `repetition_cycle_plies`：三次重复时的循环区间。
- `error`：该盘失败原因。

任务可暂停；JSON 会保留已完成对局，以及中断盘截至暂停时的 `snapshots`。恢复时使用同一任务和模型配置，从中断盘的 `initial_fen` 加最后一个 snapshot 重建局面后继续；不保存或重放 `moves`、`final_fen`。

## 生命周期

任务状态为 `running`、`paused`、`completed` 或 `failed`。已完成或失败任务不可恢复；暂停的未完成任务可以恢复。进程重启后会从 `BENCHMARK_PATH` 加载 JSON，暂停任务仍可恢复。