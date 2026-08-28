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


class ResNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, value_head: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.start_head = nn.Linear(channels * 10 * 9, 90)
        self.end_head = nn.Linear(channels * 10 * 9, 90)
        self.value_head = (
            nn.Sequential(
                nn.Linear(channels * 10 * 9, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            if value_head
            else None
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        features = self.residual_blocks(self.stem(inputs)).flatten(1)
        outputs: tuple[Tensor, ...] = (self.start_head(features), self.end_head(features))
        if self.value_head is not None:
            outputs += (self.value_head(features),)
        return outputs


class PikafishResNet(nn.Module):
    def __init__(self, channels: int = 192, blocks: int = 12) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Conv2d(channels, 90, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 10 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.residual_blocks(self.stem(inputs))
        policy_logits = self.policy_head(features).permute(0, 2, 3, 1).reshape(-1, 8100)
        return policy_logits, self.value_head(features)
