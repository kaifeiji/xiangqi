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

## 文献参考

- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404)（Silver et al., Science, 2018）：介绍基于神经网络策略/价值双头与蒙特卡洛树搜索的自博弈系统。
- [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439)（Anthony, Tian, Barber, NeurIPS, 2017）：介绍 Expert Iteration，以及深度学习模型与树搜索之间的迭代训练流程。
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)（Hinton, Vinyals, Dean, 2015）：介绍使用温度参数生成软目标的知识蒸馏方法。
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)（Haarnoja et al., ICML, 2018）：介绍最大熵强化学习和基于 Boltzmann 分布的随机策略。
- [Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://doi.org/10.1145/3394486.3409195)（McIlroy-Young et al., KDD, 2020）：研究棋类人工智能与人类棋步行为的对齐和建模。
- [Not All Samples Are Created Equal: Deep Learning with Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html)（Katharopoulos, Fleuret, ICML, 2018）：研究基于样本信息量的深度学习重要性采样方法。
- [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)（Zinkevich et al., NeurIPS, 2007）：提出反事实后悔最小化方法，建立不完全信息博弈中的迭代求解框架。
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)（Hu, Shen, Sun, CVPR, 2018）：提出 Squeeze-and-Excitation 模块，通过通道间信息建模和通道重标定增强卷积网络表示能力。
- [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29)（Kocsis, Szepesvári, ECML, 2006）：提出 UCT，将置信上界策略用于蒙特卡洛树搜索中的节点选择。
- [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/Coulom-2006.pdf)（Coulom, 2006）：研究蒙特卡洛树搜索中的选择、模拟和回传机制。
