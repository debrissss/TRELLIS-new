"""
Convert a TRELLIS training .pt state_dict checkpoint to .json + .safetensors.

This script is intentionally strict: it builds the model from the training
config, loads the .pt state_dict with strict=True, writes a TRELLIS
from_pretrained-compatible pair, and verifies the result by loading it through
trellis.models.from_pretrained.
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
        description="Convert a TRELLIS .pt state_dict checkpoint to .json + .safetensors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .pt state_dict checkpoint, e.g. denoiser_ema0.9999_step0000010.pt.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Output prefix without suffix. Writes <prefix>.json and <prefix>.safetensors.",
    )
    parser.add_argument(
        "--train_config",
        type=Path,
        required=True,
        help="Training config JSON. models[model_key] is used as the output model config.",
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
        help="Overwrite outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_json = args.output_prefix.with_suffix(".json")
    output_safetensors = args.output_prefix.with_suffix(".safetensors")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input}")
    if not args.train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {args.train_config}")
    for output in [output_json, output_safetensors]:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'safetensors'. Install project training dependencies before conversion."
        ) from e

    model_config = get_train_model_config(args.train_config, args.model_key)

    print(f"[INFO] Loading model config from {args.train_config} models.{args.model_key}", flush=True)
    model = build_model(model_config)

    print(f"[INFO] Loading PyTorch state_dict from {args.input}", flush=True)
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"{args.input} must contain a state_dict dictionary.")

    model.load_state_dict(state_dict, strict=True)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing model config to {output_json}", flush=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[INFO] Writing safetensors weights to {output_safetensors}", flush=True)
    save_file(model.state_dict(), str(output_safetensors))

    print("[INFO] Verifying output with trellis.models.from_pretrained(...)", flush=True)
    from trellis import models

    reloaded_model = models.from_pretrained(str(args.output_prefix))
    verify_model = build_model(model_config)
    verify_model.load_state_dict(reloaded_model.state_dict(), strict=True)

    print("[INFO] Conversion completed and verified.", flush=True)


if __name__ == "__main__":
    main()
