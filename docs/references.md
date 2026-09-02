# 文献与外部实现参考

## 强化学习、树搜索与蒸馏

- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404)（Silver et al., Science, 2018）：基于神经网络策略/价值双头与蒙特卡洛树搜索的自博弈系统。
- [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439)（Anthony, Tian, Barber, NeurIPS, 2017）：Expert Iteration，以及深度学习模型与树搜索之间的迭代训练流程。
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)（Hinton, Vinyals, Dean, 2015）：使用温度参数生成软目标的知识蒸馏方法。
- [Policy Distillation](https://arxiv.org/abs/1511.06295)（Rusu et al., 2015）：以软化动作目标蒸馏策略。
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)（Haarnoja et al., ICML, 2018）：最大熵强化学习和基于 Boltzmann 分布的随机策略。
- [Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://doi.org/10.1145/3394486.3409195)（McIlroy-Young et al., KDD, 2020）：棋类人工智能与人类棋步行为的对齐和建模。
- [Not All Samples Are Created Equal: Deep Learning with Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html)（Katharopoulos, Fleuret, ICML, 2018）：基于样本信息量的重要性采样。
- [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)（Zinkevich et al., NeurIPS, 2007）：反事实后悔最小化。
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)（Hu, Shen, Sun, CVPR, 2018）：通道重标定模块。
- [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29)（Kocsis, Szepesvari, ECML, 2006）：UCT。
- [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/Coulom-2006.pdf)（Coulom, 2006）：MCTS 选择、模拟和回传机制。

## Pikafish、Stockfish 与 UCI

- [Pikafish types.h](https://github.com/official-pikafish/Pikafish/blob/master/src/types.h)：mate 与普通搜索值的内部编码。
- [Pikafish score.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/score.cpp)：普通值和 mate 在 UCI 输出边界的类型转换。
- [Pikafish uci.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/uci.cpp)：`score cp`、`score mate` 和 WDL 的格式化逻辑。
- [Pikafish engine.cpp](https://github.com/official-pikafish/Pikafish/blob/master/src/engine.cpp)：引擎选项，包括 `UCI_ShowWDL`。
- [Stockfish types.h](https://github.com/official-stockfish/Stockfish/blob/master/src/types.h)、[score.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/score.cpp) 与 [uci.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp)：国际象棋引擎的对应实现；其 tablebase 特殊编码不可直接迁用到象棋训练标签。
- [UCI Protocol](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html)：UCI 协议说明与规范下载页。

## PyTorch 数据加载

- [PyTorch Data Loading](https://docs.pytorch.org/docs/stable/data.html)：map-style dataset、worker 与采样器行为。
- [DataLoader source](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)：DataLoader 实现参考。
