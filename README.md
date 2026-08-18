# 象棋模型训练项目

本项目面向消费级显卡训练中国象棋模型，使用人类棋局数据进行训练。

项目包含数据准备、ResNet 模型训练、模型评估和棋规引擎。数据准备阶段将不同来源的人类棋局统一转换为局面、走法标签和可选的终局价值标签；训练阶段使用统一的棋盘张量训练 ResNet 模型。

- Python 3.11/3.12
- PyTorch 2.x
- 输入张量：`(15, 10, 9)`
- ResNet 模型：起点/终点策略 logits

## 环境准备

```powershell
python -m pip install uv
python -m uv sync
```

## 启动

项目提供 Web 对弈服务，生产模式和开发模式使用同一个入口。

生产模式：

```powershell
uv run xiangqi-play --host 127.0.0.1 --port 8000
```

开发模式：

```powershell
uv run xiangqi-play --dev
```

开发模式访问 `http://127.0.0.1:5173`，后端 API 地址为 `http://127.0.0.1:8000`。

## 前端开发

前端源码位于 `src/frontend/`，使用 React、TypeScript 和 Vite。开发模式启动后，修改 React、TypeScript 或 CSS 文件会自动热更新。

单独检查前端构建：

```powershell
npm ci
npm run build
```

## 后端开发

后端源码位于 `src/backend/`，包含 Flask API、棋规引擎、模型推理和 player 实现。也可以直接运行 Flask 入口：

```powershell
uv run python -m backend.app --host 127.0.0.1 --port 8000
```

## 模型加载

训练完成后，将模型 checkpoint 放入 `models/` 目录。支持 `.pt`、`.pth` 和 `.ckpt` 格式；启动服务后，模型即可被加载使用。

## 测试

运行全项目测试：

```powershell
uv run pytest -q
```

修改代码后，优先运行对应测试，再运行完整测试集。

数据准备、训练、评估和测试命令见 [scripts/README.md](scripts/README.md)。
