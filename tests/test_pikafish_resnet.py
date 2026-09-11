from __future__ import annotations

import pytest


@pytest.fixture
def training_models(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from training_models import PikafishResNet, ResNet

    return ResNet, PikafishResNet


def test_pikafish_resnet_outputs_joint_policy_and_bounded_value(training_models) -> None:
    torch = pytest.importorskip("torch")
    _, PikafishResNet = training_models

    model = PikafishResNet(channels=16, blocks=2, use_se=True)
    policy_logits, value = model(torch.randn(2, 15, 10, 9))
    loss = policy_logits.square().mean() + value.square().mean()
    loss.backward()

    assert policy_logits.shape == (2, 8100)
    assert value.shape == (2, 1)
    assert torch.all(value >= -1)
    assert torch.all(value <= 1)
    assert model.policy_head.weight.grad is not None
    assert model.value_head[1].weight.grad is not None


def test_resnet_se_outputs_policy_and_value(training_models) -> None:
    torch = pytest.importorskip("torch")
    ResNet, _ = training_models

    model = ResNet(channels=16, blocks=2, value_head=True, use_se=True)
    start_logits, end_logits, value = model(torch.randn(2, 15, 10, 9))
    (start_logits.square().mean() + end_logits.square().mean() + value.square().mean()).backward()

    assert start_logits.shape == (2, 90)
    assert end_logits.shape == (2, 90)
    assert value.shape == (2, 1)
    assert model.residual_blocks[0].se.fc2.bias.grad is not None


def test_export_loader_restores_se_from_checkpoint(training_models, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    ResNet, _ = training_models
    from export_onnx import load_model

    source = ResNet(channels=16, blocks=2, value_head=True, use_se=True)
    checkpoint_path = tmp_path / "se.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "config": {"channels": 16, "blocks": 2, "use_se": True, "se_reduction": 16},
        },
        checkpoint_path,
    )

    restored, joint_policy = load_model(checkpoint_path)

    assert not joint_policy
    assert isinstance(restored.residual_blocks[0].se, torch.nn.Module)
    assert not isinstance(restored.residual_blocks[0].se, torch.nn.Identity)