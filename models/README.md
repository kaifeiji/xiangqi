# Web 模型目录

请将训练生成的模型 checkpoint 放在这里或其子目录。服务递归扫描本目录中的 `.pt`、`.pth` 和 `.ckpt` 文件，并将文件名显示为 Web 端模型选项；子目录模型的 ID 是相对于 `models/` 的路径。

`train.py` 与 `train_pikafish.py` 生成的完整 checkpoint 都包含模型 state 和 config，可被服务自动识别。服务根据 state 中的 policy head 自动区分 `ResNet` 与 `PikafishResNet`；后者使用空间式 8100-logit policy head 和 current-view 推理。发布训练结果时复制选择出的 checkpoint，例如：

```powershell
Copy-Item checkpoints\pikafish-g4096-m2048\best.pt models\pikafish-c192-b12.pt
```

重启服务后请求 `GET /api/models`，或运行 `scripts/benchmark_models.py --games 2`，可验证该 checkpoint 已被发现并可走子。

Pikafish NNUE 引擎本身不放在本目录，也不作为 PyTorch checkpoint 加载。Web 服务通过进程环境变量 `PIKAFISH_PATH` 和可选的 `PIKAFISH_NNUE_PATH` 配置它。
