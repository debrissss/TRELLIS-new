"""将完整 SS Flow ControlNet 训练权重转换为可迁移部署产物。

与通用转换器不同，源模型允许通过训练配置中的本地
``control_encoder_ckpt`` 构造；写出的部署 JSON 会删除该字段，并用完整
state_dict 携带 control encoder 权重，从而不依赖训练机路径。
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.convert_pt_to_safetensors import (
    build_model,
    get_output_paths,
    get_train_model_config,
    load_json,
    validate_model_config,
)


CONTROLNET_MODEL_NAME = "SparseStructureFlowModel_ControlNet"


def make_portable_model_config(model_config: dict) -> dict:
    """复制部署配置并移除只属于源模型构造阶段的本地 checkpoint。"""
    validate_model_config(model_config, Path("<model-config>"))
    if model_config["name"] != CONTROLNET_MODEL_NAME:
        raise ValueError(
            f"ControlNet converter requires {CONTROLNET_MODEL_NAME}, "
            f"got {model_config['name']}"
        )
    portable_config = copy.deepcopy(model_config)
    portable_config["args"].pop("control_encoder_ckpt", None)
    return portable_config


def get_deploy_model_config(
    source_model_config: dict,
    deploy_model_config: Path | None,
) -> dict:
    if deploy_model_config is None:
        return make_portable_model_config(source_model_config)
    explicit_config = load_json(deploy_model_config)
    validate_model_config(explicit_config, deploy_model_config)
    return make_portable_model_config(explicit_config)


def validate_complete_controlnet_state(model, state_dict: dict) -> None:
    """防止把仅含可训练控制分支、却缺冻结 encoder 的权重误发布。"""
    encoder_keys = {
        key for key in model.state_dict() if key.startswith("control_encoder.")
    }
    missing_encoder_keys = sorted(encoder_keys.difference(state_dict))
    if missing_encoder_keys:
        preview = ", ".join(missing_encoder_keys[:5])
        raise RuntimeError(
            "ControlNet deployment state_dict is missing frozen control encoder "
            f"weights: {preview}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a complete ControlNet .pt checkpoint to portable JSON + safetensors."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--train_config", type=Path, required=True)
    parser.add_argument("--model_key", default="denoiser")
    parser.add_argument(
        "--deploy_model_config",
        type=Path,
        help=(
            "Optional direct {name,args} JSON used for deployment. "
            "control_encoder_ckpt is always removed from the emitted JSON."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json, output_safetensors = get_output_paths(args.output_prefix)

    if not args.input.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input}")
    if not args.train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {args.train_config}")
    if args.deploy_model_config is not None and not args.deploy_model_config.is_file():
        raise FileNotFoundError(
            f"Deployment model config not found: {args.deploy_model_config}"
        )
    for output in (output_json, output_safetensors):
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )

    from safetensors.torch import load_file, save_file

    # 源配置可以保留本地 encoder 路径，仅用于还原训练时构造环境。
    source_config = get_train_model_config(args.train_config, args.model_key)
    if source_config["name"] != CONTROLNET_MODEL_NAME:
        raise ValueError(
            f"Training model must be {CONTROLNET_MODEL_NAME}, "
            f"got {source_config['name']}"
        )
    source_model = build_model(source_config)
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{args.input} must contain a state_dict dictionary")
    source_model.load_state_dict(checkpoint, strict=True)
    complete_state = source_model.state_dict()
    validate_complete_controlnet_state(source_model, complete_state)

    # 先用无本地路径的部署配置重建并严格加载，再允许写盘；显式部署配置若
    # 架构不匹配会在此处失败，而不是生成不可用包。
    deploy_config = get_deploy_model_config(
        source_config, args.deploy_model_config
    )
    deploy_model = build_model(deploy_config)
    deploy_model.load_state_dict(complete_state, strict=True)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(deploy_config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    save_file(complete_state, str(output_safetensors))

    # 最终验证只读刚写出的 portable JSON 和 safetensors，不能复用源配置，
    # 以确保产物脱离训练机的 control_encoder_ckpt 仍可独立恢复。
    written_config = load_json(output_json)
    if "control_encoder_ckpt" in written_config["args"]:
        raise RuntimeError("Portable output unexpectedly contains control_encoder_ckpt")
    verified_model = build_model(written_config)
    verified_model.load_state_dict(
        load_file(str(output_safetensors)), strict=True
    )
    print("ControlNet conversion completed and portable output verified.")


if __name__ == "__main__":
    main()
