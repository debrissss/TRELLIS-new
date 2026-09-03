#!/usr/bin/env python3
"""Run the sparse-structure decoder as an independent inference stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json
from eval.common.ss_inference import (
    latent_stats,
    load_configured_model,
    load_ss_latent,
    require_device,
    sample_output_dir,
    successful_latent_rows,
    write_stage_result,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SS VAE training config JSON")
    parser.add_argument(
        "--decoder_ckpt",
        required=True,
        help="Decoder .pt checkpoint or from_pretrained-compatible model prefix",
    )
    parser.add_argument(
        "--latent_manifest",
        type=Path,
        required=True,
        help="manifest.csv produced by ss_encoder_inference or ss_flow_inference",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def _bbox(coords: np.ndarray) -> tuple[list[int] | None, list[int] | None]:
    if coords.size == 0:
        return None, None
    return coords.min(axis=0).astype(int).tolist(), coords.max(axis=0).astype(int).tolist()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from fine_tuning.audit_ss_gt_reconstruction import (
        occupancy_to_mesh,
        write_occupancy_points,
    )

    config = load_json(args.config)
    producer_rows = successful_latent_rows(args.latent_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(args.device)
    decoder = load_configured_model(
        config,
        model_key="decoder",
        checkpoint=args.decoder_ckpt,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_index, producer_row in enumerate(producer_rows):
            sample_id = producer_row["sample_id"]
            latent_path = Path(producer_row["latent_path"])
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            occupancy_path = sample_dir / "occupancy.npz"
            coords_path = sample_dir / "coords.npz"
            points_path = sample_dir / "occupied_points.ply"
            mesh_path = sample_dir / "mesh.ply"
            logits_path = sample_dir / "logits.npz"
            stats_path = sample_dir / "stats.json"
            source_stage = producer_row.get("stage", "unknown")
            base_row = {
                "stage": "ss_decoder",
                "source_stage": source_stage,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "dataset_index": producer_row.get("dataset_index", sample_index),
                "latent_path": str(latent_path),
                "latent_key": producer_row.get("latent_key", ""),
                "condition_image_path": producer_row.get("condition_image_path", ""),
                "prepared_condition_path": producer_row.get("prepared_condition_path", ""),
                "condition_preprocessed": producer_row.get("condition_preprocessed", ""),
                "condition_features_path": producer_row.get(
                    "condition_features_path", ""
                ),
                "rng_state_path": producer_row.get("rng_state_path", ""),
                "seed": producer_row.get("seed", ""),
                "occupancy_path": str(occupancy_path.resolve()),
                "coords_path": str(coords_path.resolve()),
                "points_path": str(points_path.resolve()),
                "mesh_path": str(mesh_path.resolve()) if args.save_mesh else "",
                "logits_path": str(logits_path.resolve()) if args.save_logits else "",
                "stats_path": str(stats_path.resolve()),
                "threshold": args.threshold,
                "checkpoint": args.decoder_ckpt,
            }
            required_outputs = [occupancy_path, coords_path, points_path, stats_path]
            if args.save_mesh:
                required_outputs.append(mesh_path)
            if args.save_logits:
                required_outputs.append(logits_path)
            if args.skip_existing and all(path.is_file() for path in required_outputs):
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                latent_array, source_key = load_ss_latent(latent_path)
                latent = torch.from_numpy(latent_array).unsqueeze(0).to(device)
                logits = decoder(latent)
                logits_cpu = logits[0, 0].detach().float().cpu()
                occupancy = logits_cpu.numpy() > args.threshold
                coords = np.argwhere(occupancy).astype(np.int32, copy=False)
                bbox_min, bbox_max = _bbox(coords)

                sample_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    occupancy_path,
                    occupancy=occupancy.astype(np.uint8, copy=False),
                )
                np.savez_compressed(coords_path, coords=coords)
                write_occupancy_points(points_path, coords, occupancy.shape[0])
                if args.save_logits:
                    np.savez_compressed(logits_path, logits=logits_cpu.numpy())
                if args.save_mesh:
                    occupancy_to_mesh(occupancy).export(mesh_path)

                stats = {
                    "stage": "ss_decoder",
                    "source_stage": source_stage,
                    "sample_id": sample_id,
                    "latent_path": str(latent_path),
                    "latent_source_key": source_key,
                    "decoder_checkpoint": args.decoder_ckpt,
                    "threshold": args.threshold,
                    "latent": latent_stats(latent[0]),
                    "logits": latent_stats(logits_cpu),
                    "occupancy_shape": list(occupancy.shape),
                    "occupied_voxels": int(coords.shape[0]),
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                    "occupancy_path": str(occupancy_path.resolve()),
                    "coords_path": str(coords_path.resolve()),
                    "points_path": str(points_path.resolve()),
                    "mesh_path": str(mesh_path.resolve()) if args.save_mesh else None,
                    "logits_path": str(logits_path.resolve()) if args.save_logits else None,
                }
                write_json(stats_path, stats)
                rows.append(
                    {
                        **base_row,
                        "latent_key": source_key,
                        "occupied_voxels": int(coords.shape[0]),
                        "bbox_min": json.dumps(bbox_min),
                        "bbox_max": json.dumps(bbox_max),
                        "skipped": False,
                        "failed": False,
                        "error": "",
                    }
                )
                print(f"[{sample_index + 1}/{len(producer_rows)}] decoded {sample_id}", flush=True)
            except Exception as exc:
                failure = {
                    **base_row,
                    "occupied_voxels": "",
                    "bbox_min": "",
                    "bbox_max": "",
                    "skipped": False,
                    "failed": True,
                    "error": repr(exc),
                }
                rows.append(failure)
                print(
                    f"[{sample_index + 1}/{len(producer_rows)}] failed {sample_id}: {exc!r}",
                    flush=True,
                )
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="ss_decoder",
                        args=args,
                        rows=rows,
                        extra_summary={
                            "source_manifest": str(args.latent_manifest),
                            "threshold": args.threshold,
                        },
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="ss_decoder",
        args=args,
        rows=rows,
        extra_summary={
            "source_manifest": str(args.latent_manifest),
            "threshold": args.threshold,
            "save_mesh": bool(args.save_mesh),
            "save_logits": bool(args.save_logits),
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
