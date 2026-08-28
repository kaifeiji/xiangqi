from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from training_models import PikafishResNet, ResNet


class UnifiedModel(nn.Module):
    def __init__(self, model: nn.Module, joint_policy: bool) -> None:
        super().__init__()
        self.model = model
        self.joint_policy = joint_policy

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(inputs)
        if self.joint_policy:
            return outputs[0], outputs[1]
        start, end = outputs[:2]
        value = outputs[2] if len(outputs) > 2 else torch.zeros(
            (inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device
        )
        return start.unsqueeze(2) + end.unsqueeze(1), value


def load_model(checkpoint_path: Path) -> tuple[nn.Module, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if any(key.startswith("policy_head.") for key in state):
        channels = state["stem.0.weight"].shape[0]
        blocks = max(
            int(key.split(".")[1]) for key in state if key.startswith("residual_blocks.")
        ) + 1
        model = PikafishResNet(channels=channels, blocks=blocks)
        joint_policy = True
    else:
        channels = int(config.get("channels", state["stem.0.weight"].shape[0]))
        blocks = int(config.get("blocks", 4))
        value_head = any(key.startswith("value_head.") for key in state)
        model = ResNet(channels=channels, blocks=blocks, value_head=value_head)
        joint_policy = False
    model.load_state_dict(state)
    model.eval()
    return model, joint_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Xiangqi PT checkpoint to unified ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model, joint_policy = load_model(args.checkpoint)
    unified = UnifiedModel(model, joint_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 15, 10, 9, dtype=torch.float32)
    torch.onnx.export(
        unified,
        dummy,
        args.output,
        input_names=["board"],
        output_names=["move_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "move_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
