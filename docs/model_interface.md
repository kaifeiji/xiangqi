# 模型接口设计

本文档定义训练模型、推理模块和棋盘编码之间的接口边界。

## 1. 模型接口

模型输入为批量棋盘张量：

```text
shape: (batch_size, 15, 10, 9)
dtype: float32
```

策略基线模型输出两个未归一化的 logits：

```text
start_logits: (batch_size, 90)
end_logits:   (batch_size, 90)
```

其中每个类别对应棋盘上的一个格子，索引范围为 `0~89`。起点头预测起始格，终点头预测目标格。

15 个输入通道由红方 7 个棋子通道、黑方 7 个棋子通道和当前行棋方通道组成。

模型 `forward` 不执行 `softmax`，训练时直接将 logits 传给 `CrossEntropyLoss`。

```python
start_logits, end_logits = model(board_batch)
loss = start_loss(start_logits, start_targets) + end_loss(end_logits, end_targets)
```

启用价值头时，模型额外输出当前行棋方视角的价值：

```text
value: (batch_size,)
range: [-1, 1]
```

```python
start_logits, end_logits, value = model(board_batch)
policy_loss = start_loss(start_logits, start_targets) + end_loss(end_logits, end_targets)
loss = policy_loss + 0.5 * mse_loss(value, value_targets)
```

## 2. 训练数据接口

单条样本包含：

```python
{
    "position": Tensor,       # (15, 10, 9)
    "start_index": int,       # 0~89
    "end_index": int,         # 0~89
    "game_id": str,
    "ply": int,
}
```

Dataset 返回训练所需的三个字段：

```python
board, start_index, end_index
```

启用价值头时，数据分片额外包含 `values.npy`，Dataset 返回：

```python
board, start_index, end_index, value
```

`value` 按当前行棋方计算：己方最终获胜为 `+1`，和棋为 `0`，己方最终失败为 `-1`。已有 memory-mapped 数据可通过 `prepare_data.py --add-values` 添加该数组。

棋谱来源、局号和 ply 保存在元数据中，用于追溯、去重和错误定位，不随每个批次传入 GPU。

## 3. 推理接口

推理模块对外返回统一的候选走法结果，而不是直接暴露两个 logits：

```python
from dataclasses import dataclass


@dataclass
class MovePrediction:
    iccs: str
    start_index: int
    end_index: int
    score: float
    start_probability: float
    end_probability: float
```

推荐接口：

```python
def predict_move(
    model,
    position,
    *,
    top_k: int = 8,
    legal_moves: list[tuple[int, int]] | None = None,
) -> list[MovePrediction]:
    ...
```

调用方传入 `(15, 10, 9)` 或带 batch 维度的棋盘张量，返回按分数降序排列的候选走法。`top_k` 用于限制起点和终点候选数量。

## 4. 合法走法掩码与双头输出

第一阶段训练不使用联合动作 mask，起点和终点头分别使用真实 ICCS 标签计算交叉熵。这样先建立可复现的监督学习基线。

数据准备阶段不重新验证棋谱合法性。推理阶段仍必须由棋规模块生成当前局面的合法走法集合，将合法 `(start_index, end_index)` 作为动作 mask；非法组合不参与排序或概率归一化。由于终点是否合法依赖起点，不能对两个 90 类头使用彼此独立的全局 mask。

后续优化可实现条件终点 mask：对每个候选起点，只保留该起点对应的合法终点，近似建模 `P(end | start, position)`。

不能分别取起点头和终点头的最大值后直接组成走法，因为组合结果可能不合法。推理流程如下：

1. 生成当前局面的全部合法 `(start_index, end_index)`。
2. 对每一条合法走法计算起点和终点 logits 的联合分数。
3. 非法组合不参与排序和概率归一化。
4. 按联合分数排序，返回前 `top_k` 条并转换为 ICCS 字符串。

实现不得先截断两个独立头的 Top-K 再过滤，否则真实走法可能因起点或终点未进入局部 Top-K 而被错误丢弃。若需要加速，可使用自适应候选上限，但必须保留“全合法走法评分”作为正确性基线。

基础评分定义为：

```text
score = log_softmax(start_logits)[start]
      + log_softmax(end_logits)[end]
```

价值头已经可用于监督训练；后续接入 MCTS 时可直接复用第三个价值输出：

```text
value: (batch_size,)
```

价值使用当前行棋方视角，胜为 `+1`、和为 `0`、负为 `-1`。扩展后的模型接口为：

```python
start_logits, end_logits, value = model(board_batch)
```

策略基线仍可只返回两个 logits。搜索模块可提供：

```python
def evaluate_position(
    model,
    position,
    legal_moves: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    ...
```

该接口返回每个合法 `(start_index, end_index)` 的模型概率，搜索模块不需要了解神经网络的双头细节。

## 5. 棋盘和 ICCS 转换接口

以下函数必须共用同一套坐标约定：

```python
def encode_fen(fen: str):
    """将 FEN 转为 (15, 10, 9) 的 float32 棋盘张量。"""


def iccs_to_indices(move: str) -> tuple[int, int]:
    """将 C3-C4 转为起点和终点索引。"""


def indices_to_iccs(start_index: int, end_index: int) -> str:
    """将起点、终点索引转为 C3-C4。"""


def apply_move(position, start_index: int, end_index: int):
    """执行走法并返回新局面。"""


def generate_legal_moves(position) -> list[tuple[int, int]]:
    """返回当前局面的合法起点、终点组合。"""
```

ICCS 的列 `A-I` 映射到 `0-8`，行号为 `0-9`。FEN 行索引与 ICCS 行号的翻转关系必须通过单元测试固定，避免训练标签和推理结果整体错位。

## 6. 调用链

```text
FEN
  -> encode_fen()
    -> (1, 15, 10, 9)
  -> model.forward()
    -> (start_logits, end_logits) 或 (start_logits, end_logits, value)
  -> Top-K 起点/终点组合
  -> 合法走法过滤
  -> ICCS 走法，例如 C3-C4
```
