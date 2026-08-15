from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


class TinyResNet(nn.Module):
    """Four-block, two-head network for Xiangqi move prediction."""

    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(blocks))
        )
        self.start_head = nn.Linear(channels * 10 * 9, 90)
        self.end_head = nn.Linear(channels * 10 * 9, 90)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.residual_blocks(self.stem(inputs)).flatten(1)
        return self.start_head(features), self.end_head(features)
