#!/usr/bin/env python3
"""Run the structured-latent encoder as an independent inference stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json
from eval.common.slat_inference import (
    require_nonempty_unique_samples,
    resolve_slat_encoder_inputs,
    save_slat_latent,
    slat_stats,
    write_slat_input_manifest,
)
from eval.common.ss_inference import (
    load_configured_model,
    require_device,
    sample_output_dir,
    write_stage_result,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SLat encoder/VAE config JSON")
    parser.add_argument(
        "--encoder_ckpt",
        required=True,
        help="Encoder .pt checkpoint or from_pretrained-compatible model prefix",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--data_dir",
        type=Path,
        help="Dataset root containing metadata.csv and features/",
    )
    inputs.add_argument(
        "--input_manifest",
        type=Path,
        help="Fixed manifest containing sample_id, data_dir, and feature_path",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feature_model", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=16, help="<=0 selects all valid samples")
    parser.add_argument("--indices", default=None, help="Comma-separated indices in the valid-sample list")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    import torch

    # Reuse the current fixed-view SLat evaluation loader. It implements the
    # same feature downsampling and SparseTensor construction as the dataset.
    from eval.common.impl.slat_encoder_gs_decoder_reconstruction_impl import (
        load_sparse_feature,
    )
    from eval.ss_flow_inference import _set_seed

    config = load_json(args.config)
    dataset_args = config.get("dataset", {}).get("args", {})
    feature_model = args.feature_model or dataset_args.get("model", "dinov2_vitl14_reg")
    resolution = int(
        args.resolution
        if args.resolution is not None
        else dataset_args.get("resolution", 64)
    )
    min_aesthetic_score = dataset_args.get("min_aesthetic_score")
    max_num_voxels = dataset_args.get("max_num_voxels")
    records = resolve_slat_encoder_inputs(
        data_dir=args.data_dir,
        input_manifest=args.input_manifest,
        feature_model=feature_model,
        num_samples=args.num_samples,
        seed=args.seed,
        indices=args.indices,
        min_aesthetic_score=(
            float(min_aesthetic_score) if min_aesthetic_score is not None else None
        ),
        max_num_voxels=(
            int(max_num_voxels) if max_num_voxels is not None else None
        ),
    )
    require_nonempty_unique_samples(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_slat_input_manifest(args.output_dir / "selected_samples.csv", records)

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
            data_dir = Path(record["data_dir"])
            feature_path = Path(record["feature_path"])
            record_feature_model = record["feature_model"]
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            latent_path = sample_dir / "latent.npz"
            stats_path = sample_dir / "stats.json"
            base_row = {
                "stage": "slat_encoder",
                "sample_id": sample_id,
                "sample_index": sample_index,
                "dataset_index": record["dataset_index"],
                "data_dir": str(data_dir),
                "feature_model": record_feature_model,
                "feature_path": str(feature_path),
                "condition_image_path": record.get("condition_image_path", ""),
                "latent_path": str(latent_path.resolve()),
                "latent_keys": "coords,feats,mean,logvar",
                "latent_domain": "decoder_ready",
                "checkpoint": args.encoder_ckpt,
            }
            if args.skip_existing and latent_path.is_file() and stats_path.is_file():
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                _set_seed(args.seed + sample_index)
                sparse_features = load_sparse_feature(
                    data_dir,
                    sample_id,
                    record_feature_model,
                    resolution,
                ).to(device)
                z, mean, logvar = encoder(
                    sparse_features,
                    sample_posterior=args.sample_posterior,
                    return_raw=True,
                )
                coords = z.coords[:, 1:]
                save_slat_latent(
                    latent_path,
                    coords=coords,
                    feats=z.feats,
                    mean=mean,
                    logvar=logvar,
                )
                stats = {
                    "stage": "slat_encoder",
                    "sample_id": sample_id,
                    "feature_path": str(feature_path),
                    "feature_model": record_feature_model,
                    "encoder_checkpoint": args.encoder_ckpt,
                    "resolution": resolution,
                    "sample_posterior": bool(args.sample_posterior),
                    "latent_domain": "decoder_ready",
                    "z": slat_stats(coords, z.feats),
                    "mean": slat_stats(coords, mean),
                    "logvar": slat_stats(coords, logvar),
                }
                write_json(stats_path, stats)
                rows.append({**base_row, "skipped": False, "failed": False, "error": ""})
                print(f"[{sample_index + 1}/{len(records)}] encoded {sample_id}", flush=True)
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "skipped": False,
                        "failed": True,
                        "error": repr(exc),
                    }
                )
                print(
                    f"[{sample_index + 1}/{len(records)}] failed {sample_id}: {exc!r}",
                    flush=True,
                )
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="slat_encoder",
                        args=args,
                        rows=rows,
                        extra_summary={
                            "feature_model": feature_model,
                            "resolution": resolution,
                            "latent_domain": "decoder_ready",
                        },
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="slat_encoder",
        args=args,
        rows=rows,
        extra_summary={
            "feature_model": feature_model,
            "resolution": resolution,
            "latent_domain": "decoder_ready",
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
