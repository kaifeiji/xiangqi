# Web 模型目录

请将训练生成的模型 checkpoint 放在这里或其子目录。服务递归扫描本目录中的 `.pt`、`.pth` 和 `.ckpt` 文件，并将文件名显示为 Web 端模型选项；子目录模型的 ID 是相对于 `models/` 的路径。

优先使用 `scripts/train.py` 生成的 checkpoint，因为其中包含模型重建所需的 `model` 和 `config` 信息。`config.channels`、`config.blocks`、`config.value_head` 以及数据视角配置会影响模型加载；value head 和 current-view 模型建议在文件名或目录名中显式标记。

Pikafish 不放在本目录，也不作为 PyTorch checkpoint 加载。它通过 `.env.local` 的 `PIKAFISH_PATH` 和可选的 `PIKAFISH_NNUE_PATH` 单独配置。
