"""Prepare tiny FaceScape SLat overfit datasets and training configs.

This script is newly added for the SLat single-sample overfit diagnostic. It
intentionally mirrors the SS overfit setup while keeping SS preparation code
untouched: same selected neutral sample, same tiny-dataloader safeguards, and
the same save/sample/log cadence.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRAIN_ROOT = PROJECT_ROOT / "datasets" / "Facescape" / "train"
BASE_CONFIG = PROJECT_ROOT / "configs" / "generation" / "slat_flow_img_dit_L_64l8p2_fp16_finetune_facescape.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "Facescape"
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "configs" / "generation" / "overfit"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
DEFAULT_COUNT = 1


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_metadata(metadata_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_neutral_row(row: dict[str, str]) -> bool:
    return row.get("captions", "").endswith("_1_neutral")


def has_required_assets(row: dict[str, str], source_root: Path) -> bool:
    sha = row.get("sha256", "")
    return (
        bool(sha)
        and (source_root / "renders_cond" / sha).is_dir()
        and (source_root / "latents" / SLAT_LATENT_MODEL / f"{sha}.npz").is_file()
    )


def select_neutral_rows(rows: list[dict[str, str]], source_root: Path, count: int) -> list[dict[str, str]]:
    selected = [row for row in rows if is_neutral_row(row) and has_required_assets(row, source_root)]
    return selected[:count]


def replace_path_with_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare_subset(source_root: Path, subset_root: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows selected for subset {subset_root}")

    source_metadata = source_root / "metadata.csv"
    if source_metadata.is_file():
        fieldnames = list(csv.DictReader(source_metadata.open(newline="", encoding="utf-8")).fieldnames or rows[0].keys())
    else:
        fieldnames = list(rows[0].keys())
    write_metadata(subset_root / "metadata.csv", rows, fieldnames)

    for row in rows:
        sha = row["sha256"]
        # New SLat audit path: reuse the exact same normal-image conditioning directory as SS overfit.
        replace_path_with_symlink(source_root / "renders_cond" / sha, subset_root / "renders_cond" / sha)
        # New SLat audit path: add the structured latent target required by ImageConditionedSLat.
        replace_path_with_symlink(
            source_root / "latents" / SLAT_LATENT_MODEL / f"{sha}.npz",
            subset_root / "latents" / SLAT_LATENT_MODEL / f"{sha}.npz",
        )


def build_overfit_config(base_config: dict, max_steps: int, sample_count: int = 1) -> dict:
    config = deepcopy(base_config)
    trainer_args = config["trainer"]["args"]

    # New SLat overfit diagnostic settings: mirror SS overfit tiny-data behavior
    # so the experiment tests SLat learnability rather than DataLoader edge cases.
    trainer_args["max_steps"] = max_steps
    trainer_args["batch_size_per_gpu"] = max(1, min(4, sample_count))
    trainer_args["batch_split"] = 1
    trainer_args["dataloader_num_workers"] = 0
    trainer_args["dataloader_drop_last"] = False
    trainer_args["dataloader_persistent_workers"] = False
    trainer_args["prefetch_data"] = False
    trainer_args["i_save"] = 500
    trainer_args["i_sample"] = 500
    trainer_args["i_log"] = 10
    trainer_args["i_print"] = 10
    return config


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def write_run_script(script_path: Path, config_path: Path, data_dir: Path, output_dir: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535
CONDA_ENV="${{CONDA_ENV:-trellis5090}}"
CONDA_BASE="${{CONDA_BASE:-/root/autodl-tmp/mamba_envs}}"
PYTHON="${{PYTHON:-/root/autodl-tmp/mamba_envs/trellis5090/bin/python}}"
if [ -z "${{OMP_NUM_THREADS:-}}" ] || [ "${{OMP_NUM_THREADS}}" = "0" ]; then
  export OMP_NUM_THREADS=8
fi
if [ -z "${{MKL_NUM_THREADS:-}}" ] || [ "${{MKL_NUM_THREADS}}" = "0" ]; then
  export MKL_NUM_THREADS=8
fi
export ATTN_BACKEND="${{ATTN_BACKEND:-sdpa}}"
export SPARSE_ATTN_BACKEND="${{SPARSE_ATTN_BACKEND:-flash_attn}}"
export SPCONV_ALGO="${{SPCONV_ALGO:-native}}"

if [ -f "${{CONDA_BASE}}/etc/profile.d/conda.sh" ]; then
  source "${{CONDA_BASE}}/etc/profile.d/conda.sh"
  conda activate "${{CONDA_ENV}}"
fi

cd {PROJECT_ROOT}
"${{PYTHON}}" train.py \\
  --config {config_path} \\
  --data_dir {data_dir} \\
  --output_dir {output_dir} \\
  --num_gpus "${{NUM_GPUS:-1}}" \\
  --ckpt none \\
  "$@"
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SLat single-sample overfit dataset and config.")
    parser.add_argument("--source_train_root", type=Path, default=SOURCE_TRAIN_ROOT)
    parser.add_argument("--base_config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config_root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_metadata(args.source_train_root / "metadata.csv")
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    selected = select_neutral_rows(rows, args.source_train_root, args.count)

    name = f"overfit_{args.count}"
    subset_root = args.dataset_root / name
    config_path = args.config_root / f"slat_flow_facescape_{name}.json"
    output_dir = args.output_root / f"slat_flow_facescape_{name}"
    script_path = PROJECT_ROOT / "fine_tuning" / f"train_slat_flow_facescape_{name}.sh"

    print(f"Source train root: {args.source_train_root}")
    print(f"- {name}: selected={len(selected)}, data={subset_root}, config={config_path}, output={output_dir}")
    if args.dry_run:
        return 0

    prepare_subset(args.source_train_root, subset_root, selected)
    write_json(config_path, build_overfit_config(base_config, args.max_steps, sample_count=args.count))
    write_run_script(script_path, config_path, subset_root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
