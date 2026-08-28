# Web 模型目录

请将导出的 ONNX 模型放在这里或其子目录。服务只扫描 `.onnx` 文件，并将文件名显示为 Web 端模型选项；子目录模型的 ID 是相对于 `models/` 的路径。

训练生成的 `.pt` checkpoint 不能直接被 Web 服务加载，需要先转换为 ONNX。导出模型统一输出 `move_logits` 和 `value`，由 Rust engine 内部完成棋盘编码、ONNX 推理和 MCTS。示例：

```powershell
uv run python scripts/export_onnx.py checkpoints\pikafish-g4096-m2048\best.pt models\resnet-c192-b12-pikafish.onnx
```

重启服务后请求 `GET /api/models`，或运行 `scripts/benchmark_models.py --games 2`，可验证该 checkpoint 已被发现并可走子。

Pikafish NNUE 引擎本身不放在本目录，也不作为模型文件加载。Web 服务通过进程环境变量 `PIKAFISH_PATH` 和可选的 `PIKAFISH_NNUE_PATH` 配置它。
