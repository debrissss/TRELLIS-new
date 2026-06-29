"""
Convert a local TRELLIS .json + .safetensors model checkpoint to a .pt state_dict.

This script is intentionally strict: it validates the safetensors weights against
the model built from the training config before writing a .pt file for
trainer.args.finetune_ckpt.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_model_config(config: dict, source: Path) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    if "name" not in config or "args" not in config:
        raise ValueError(f"{source} must contain 'name' and 'args'.")
    if not isinstance(config["name"], str):
        raise ValueError(f"{source} field 'name' must be a string.")
    if not isinstance(config["args"], dict):
        raise ValueError(f"{source} field 'args' must be an object.")


def get_train_model_config(train_config: Path, model_key: str) -> dict:
    config = load_json(train_config)
    try:
        model_config = config["models"][model_key]
    except KeyError as e:
        raise KeyError(f"{train_config} does not contain models.{model_key}.") from e
    validate_model_config(model_config, train_config)
    return model_config


def build_model(model_config: dict):
    from trellis import models

    model_cls = getattr(models, model_config["name"])
    return model_cls(**model_config["args"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local TRELLIS .safetensors weights to a .pt state_dict."
    )
    parser.add_argument(
        "--model_prefix",
        type=Path,
        required=True,
        help="Local checkpoint prefix without suffix; requires <prefix>.json and <prefix>.safetensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt state_dict path for trainer.args.finetune_ckpt.",
    )
    parser.add_argument(
        "--train_config",
        type=Path,
        default=None,
        help="Training config JSON. If provided, models[model_key] is used to build the validation model.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="denoiser",
        help="Model key under train_config.models. Defaults to denoiser.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_json = args.model_prefix.with_suffix(".json")
    model_safetensors = args.model_prefix.with_suffix(".safetensors")

    if not model_json.is_file():
        raise FileNotFoundError(f"Model config not found: {model_json}")
    if not model_safetensors.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_safetensors}")
    if args.train_config is not None and not args.train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {args.train_config}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    source_model_config = load_json(model_json)
    validate_model_config(source_model_config, model_json)

    if args.train_config is not None:
        model_config = get_train_model_config(args.train_config, args.model_key)
        config_source = f"{args.train_config} models.{args.model_key}"
    else:
        model_config = source_model_config
        config_source = str(model_json)

    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'safetensors'. Install project training dependencies before conversion."
        ) from e

    print(f"[INFO] Loading model config from {config_source}", flush=True)
    model = build_model(model_config)

    print(f"[INFO] Loading safetensors weights from {model_safetensors}", flush=True)
    state_dict = load_file(str(model_safetensors), device="cpu")
    model.load_state_dict(state_dict, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving PyTorch state_dict to {args.output}", flush=True)
    torch.save(model.state_dict(), args.output)

    print("[INFO] Verifying saved .pt with torch.load(..., weights_only=True)", flush=True)
    reloaded_state_dict = torch.load(args.output, map_location="cpu", weights_only=True)
    verify_model = build_model(model_config)
    verify_model.load_state_dict(reloaded_state_dict, strict=True)

    print("[INFO] Conversion completed and verified.", flush=True)


if __name__ == "__main__":
    main()
