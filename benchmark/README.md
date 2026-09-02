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
- `moves`：ICCS 复盘记录，用于排障。
- `final_fen`：终局 FEN。
- `repetition_cycle_plies`：三次重复时的循环区间。
- `error`：该盘失败原因。

前端卡片只展示摘要，不显示完整 `moves` 字符串，避免超长文本污染界面。需要排查异常对局时直接打开 JSON 文件。

## 前端刷新

Benchmark 页面只在存在运行中任务时轮询，轮询间隔为 5 秒。列表按 `started_at_ms` 倒序显示，最新任务在最上方。