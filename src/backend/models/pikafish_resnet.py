from __future__ import annotations

import torch
from torch import Tensor, nn

from .resnet import ResidualBlock


class PikafishResNet(nn.Module):
    """ResNet with a joint Xiangqi action policy and bounded value head."""

    def __init__(self, channels: int = 192, blocks: int = 12) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(blocks))
        )
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