#!/usr/bin/env python3
"""Run the sparse-structure encoder as an independent inference stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json
from eval.common.ss_inference import (
    latent_stats,
    load_configured_model,
    load_voxel_grid,
    require_device,
    require_nonempty_unique_samples,
    resolve_input_records,
    sample_output_dir,
    save_ss_latent,
    write_input_manifest,
    write_stage_result,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SS VAE training config JSON")
    parser.add_argument(
        "--encoder_ckpt",
        required=True,
        help="Encoder .pt checkpoint or from_pretrained-compatible model prefix",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data_dir", type=Path, help="Dataset root containing metadata.csv and voxels/")
    inputs.add_argument("--input_manifest", type=Path, help="Fixed selected_samples.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=16, help="<=0 selects all valid samples")
    parser.add_argument("--indices", default=None, help="Comma-separated indices in the valid-sample list")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = load_json(args.config)
    resolution = int(
        args.resolution
        if args.resolution is not None
        else config.get("dataset", {}).get("args", {}).get("resolution", 64)
    )
    records = resolve_input_records(
        data_dir=args.data_dir,
        input_manifest=args.input_manifest,
        num_samples=args.num_samples,
        seed=args.seed,
        indices=args.indices,
        require_voxel=True,
        require_condition=False,
    )
    require_nonempty_unique_samples(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_input_manifest(args.output_dir / "selected_samples.csv", records)

    device = require_device(args.device)
    encoder = load_configured_model(
        config,
        model_key="encoder",
        checkpoint=args.encoder_ckpt,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_index, record in enumerate(records):
            sample_id = record["sample_id"]
            voxel_path = Path(record["voxel_path"])
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            latent_path = sample_dir / "latent.npz"
            stats_path = sample_dir / "stats.json"
            base_row = {
                "stage": "ss_encoder",
                "sample_id": sample_id,
                "sample_index": sample_index,
                "dataset_index": record["dataset_index"],
                "voxel_path": str(voxel_path),
                "condition_image_path": record.get("condition_image_path", ""),
                "latent_path": str(latent_path.resolve()),
                "latent_key": "z_s",
                "checkpoint": args.encoder_ckpt,
            }
            if args.skip_existing and latent_path.is_file() and stats_path.is_file():
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                voxel = load_voxel_grid(voxel_path, resolution).unsqueeze(0).float().to(device)
                z_s, mean, logvar = encoder(
                    voxel,
                    sample_posterior=args.sample_posterior,
                    return_raw=True,
                )
                save_ss_latent(
                    latent_path,
                    z_s[0],
                    mean=mean[0],
                    logvar=logvar[0],
                )
                stats = {
                    "stage": "ss_encoder",
                    "sample_id": sample_id,
                    "voxel_path": str(voxel_path),
                    "encoder_checkpoint": args.encoder_ckpt,
                    "resolution": resolution,
                    "sample_posterior": bool(args.sample_posterior),
                    "z_s": latent_stats(z_s[0]),
                    "mean": latent_stats(mean[0]),
                    "logvar": latent_stats(logvar[0]),
                }
                write_json(stats_path, stats)
                rows.append({**base_row, "skipped": False, "failed": False, "error": ""})
                print(f"[{sample_index + 1}/{len(records)}] encoded {sample_id}", flush=True)
            except Exception as exc:
                failure = {
                    **base_row,
                    "skipped": False,
                    "failed": True,
                    "error": repr(exc),
                }
                rows.append(failure)
                print(f"[{sample_index + 1}/{len(records)}] failed {sample_id}: {exc!r}", flush=True)
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="ss_encoder",
                        args=args,
                        rows=rows,
                        extra_summary={"resolution": resolution},
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="ss_encoder",
        args=args,
        rows=rows,
        extra_summary={
            "resolution": resolution,
            "latent_model": config.get("dataset", {}).get("args", {}).get("latent_model"),
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
