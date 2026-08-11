"""Build a complete SS Flow ControlNet checkpoint from frozen base weights."""

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trellis import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation/ss_flow_finetune_ControlNet.json",
        help="ControlNet training config.",
    )
    parser.add_argument(
        "--base-flow-ckpt",
        required=True,
        help="Base SparseStructureFlowModel .pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the complete ControlNet .pt checkpoint.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)
    model_config = config["models"]["denoiser"]
    if model_config["name"] != "SparseStructureFlowModel_ControlNet":
        raise ValueError(
            "The config denoiser must be SparseStructureFlowModel_ControlNet"
        )

    # 先按 ControlNet 配置构造新增模块（包括冻结 SS Encoder 和零注入层）。
    model = getattr(models, model_config["name"])(**model_config["args"])
    base_state = torch.load(
        args.base_flow_ckpt,
        map_location="cpu",
        weights_only=True,
    )
    # 模型的兼容加载逻辑会把原 flow 前 N 层复制到 control_blocks，
    # 再保留零初始化投影，最终得到可独立恢复的完整 ControlNet state_dict。
    model.load_state_dict(base_state)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Saved complete ControlNet checkpoint: {args.output}")
    print(f"Parameters: {total:,} total, {trainable:,} trainable")


if __name__ == "__main__":
    main()
