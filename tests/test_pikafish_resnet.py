from __future__ import annotations

import pytest


def test_pikafish_resnet_outputs_joint_policy_and_bounded_value() -> None:
    torch = pytest.importorskip("torch")
    from backend.models import PikafishResNet

    model = PikafishResNet(channels=16, blocks=2)
    policy_logits, value = model(torch.randn(2, 15, 10, 9))
    loss = policy_logits.square().mean() + value.square().mean()
    loss.backward()

    assert policy_logits.shape == (2, 8100)
    assert value.shape == (2, 1)
    assert torch.all(value >= -1)
    assert torch.all(value <= 1)
    assert model.policy_head.weight.grad is not None
    assert model.value_head[1].weight.grad is not None