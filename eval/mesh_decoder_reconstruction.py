#!/usr/bin/env python3
"""Evaluate mesh decoder checkpoints with Stable3DGen-aligned mesh export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mesh_geometry_metrics import compare_meshes, load_mesh
from eval.stable3dgen_mesh_export import (
    build_stable3dgen_mesh_decoder,
    decode_latent_to_mesh_result,
    export_stable3dgen_mesh,
    load_decoder_checkpoint,
    load_json,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SLat mesh decoder checkpoints on fixed FaceScape latents."
    )
    parser.add_argument("--config", type=Path, required=True, help="Mesh decoder training config JSON")
    parser.add_argument("--data_dir", type=Path, required=True, help="Dataset split directory containing metadata.csv")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV path")
    parser.add_argument("--latent_model", required=True, help="Name under data_dir/latents/<latent_model>")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True, help="Decoder .pt checkpoints")
    parser.add_argument("--output_dir", type=Path, required=True, help="Evaluation output directory")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--point_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing_meshes", action="store_true")
    parser.add_argument("--require_all_samples", action="store_true")
    return parser.parse_args(argv)


def checkpoint_tag(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("decoder_", "")
    stem = stem.replace("step", "step")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "checkpoint"


def read_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {metadata_path}")
    if "sha256" not in rows[0]:
        raise KeyError(f"metadata.csv must contain sha256 column: {metadata_path}")
    return rows


def select_rows(data_dir: Path, rows: list[dict[str, str]], latent_model: str, num_samples: int) -> list[dict[str, str]]:
    latent_column = f"latent_{latent_model}"
    selected = []
    missing = []
    for row in rows:
        sha = row["sha256"]
        if latent_column in row and str(row[latent_column]).lower() not in {"true", "1", "yes"}:
            continue
        latent_path = data_dir / "latents" / latent_model / f"{sha}.npz"
        mesh_path = data_dir / "renders" / sha / "mesh.ply"
        if latent_path.is_file() and mesh_path.is_file():
            selected.append(row)
        else:
            missing.append((sha, str(latent_path), str(mesh_path)))
        if len(selected) >= num_samples:
            break
    if len(selected) < num_samples:
        raise RuntimeError(
            f"Only found {len(selected)} valid eval samples, requested {num_samples}. "
            f"First missing entries: {missing[:5]}"
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], failures: list[dict[str, Any]], requested: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_samples": requested,
        "successful_samples": len(rows),
        "failed_samples": len(failures),
        "success_rate": len(rows) / requested if requested else 0.0,
    }
    numeric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in {"sample_index"}
    })
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def evaluate_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    rows: list[dict[str, str]],
    data_dir: Path,
    latent_model: str,
    output_dir: Path,
    point_samples: int,
    seed: int,
    device: torch.device,
    skip_existing_meshes: bool,
    require_all_samples: bool,
) -> dict[str, Any]:
    tag = checkpoint_tag(checkpoint)
    print(f"\n=== Evaluating {tag}: {checkpoint} ===", flush=True)
    decoder = build_stable3dgen_mesh_decoder(config, device)
    load_decoder_checkpoint(decoder, checkpoint, device)
    print("Strict checkpoint load: OK", flush=True)

    per_sample: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    mesh_dir = output_dir / "meshes" / tag

    for sample_index, row in enumerate(rows):
        sha = row["sha256"]
        latent_path = data_dir / "latents" / latent_model / f"{sha}.npz"
        gt_mesh_path = data_dir / "renders" / sha / "mesh.ply"
        pred_mesh_path = mesh_dir / f"{sha}.ply"
        try:
            if skip_existing_meshes and pred_mesh_path.is_file():
                pred_mesh = load_mesh(pred_mesh_path)
            else:
                mesh_result = decode_latent_to_mesh_result(decoder, latent_path, device)
                pred_mesh = export_stable3dgen_mesh(mesh_result, pred_mesh_path)
            gt_mesh = load_mesh(gt_mesh_path)
            metrics = compare_meshes(
                pred_mesh,
                gt_mesh,
                point_samples=point_samples,
                seed=seed + sample_index * 1009,
            )
            metrics.update(
                {
                    "checkpoint": tag,
                    "checkpoint_path": str(checkpoint),
                    "sample_index": sample_index,
                    "sha256": sha,
                    "pred_mesh_path": str(pred_mesh_path),
                    "gt_mesh_path": str(gt_mesh_path),
                }
            )
            per_sample.append(metrics)
            print(f"[{sample_index + 1}/{len(rows)}] OK {sha}", flush=True)
        except Exception as exc:
            failure = {
                "checkpoint": tag,
                "sample_index": sample_index,
                "sha256": sha,
                "error": repr(exc),
                "latent_path": str(latent_path),
                "gt_mesh_path": str(gt_mesh_path),
                "pred_mesh_path": str(pred_mesh_path),
            }
            failures.append(failure)
            print(f"[{sample_index + 1}/{len(rows)}] FAIL {sha}: {exc!r}", flush=True)
            if require_all_samples:
                raise

    metrics_dir = output_dir / "metrics"
    failures_dir = output_dir / "failures"
    write_csv(metrics_dir / f"{tag}_per_sample.csv", per_sample)
    failures_dir.mkdir(parents=True, exist_ok=True)
    (failures_dir / f"{tag}_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = summarize(per_sample, failures, len(rows))
    summary.update({"checkpoint": tag, "checkpoint_path": str(checkpoint)})
    (metrics_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_json(args.config)
    metadata_path = args.metadata or args.data_dir / "metadata.csv"
    rows = select_rows(args.data_dir, read_metadata_rows(metadata_path), args.latent_model, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_samples.json").write_text(
        json.dumps([row["sha256"] for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    summaries = []
    for checkpoint in args.checkpoints:
        summaries.append(
            evaluate_checkpoint(
                config=config,
                checkpoint=checkpoint,
                rows=rows,
                data_dir=args.data_dir,
                latent_model=args.latent_model,
                output_dir=args.output_dir,
                point_samples=args.point_samples,
                seed=args.seed,
                device=device,
                skip_existing_meshes=args.skip_existing_meshes,
                require_all_samples=args.require_all_samples,
            )
        )
    write_csv(args.output_dir / "metrics" / "all_checkpoints_summary.csv", summaries)
    print(f"\nWrote mesh decoder eval results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
