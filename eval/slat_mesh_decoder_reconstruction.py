#!/usr/bin/env python3
"""Evaluate SLat mesh decoder checkpoint geometry reconstruction."""

# 中文说明：
# 评估 SLat mesh decoder checkpoint 的几何重建能力。
# 输入是已经编码好的 SLat latent、mesh decoder checkpoint 和 GT mesh。
# 输出包括预测 PLY、逐样本 Chamfer/F-score/normal consistency 等几何指标，以及多 checkpoint 汇总。

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

from eval.common.dataset import TRUE_VALUES, latent_path, read_metadata_rows, render_mesh_path
from eval.common.io import safe_tag, write_csv, write_json
from eval.common.io import load_json
from eval.common.summary import merge_summary_files


def checkpoint_tag(path: Path) -> str:
    return safe_tag(path.stem.replace("decoder_", ""))


def select_eval_rows(
    data_dir: Path,
    rows: list[dict[str, str]],
    latent_model: str,
    num_samples: int,
    *,
    require_full_count: bool = True,
) -> list[dict[str, str]]:
    latent_column = f"latent_{latent_model}"
    selected: list[dict[str, str]] = []
    missing: list[tuple[str, str, str]] = []
    for row in rows:
        sha = row["sha256"]
        if latent_column in row and str(row[latent_column]).lower() not in TRUE_VALUES:
            continue
        lp = latent_path(data_dir, latent_model, sha)
        mp = render_mesh_path(data_dir, sha)
        if lp.is_file() and mp.is_file():
            selected.append(row)
        else:
            missing.append((sha, str(lp), str(mp)))
        if len(selected) >= num_samples:
            break
    if require_full_count and len(selected) < num_samples:
        raise RuntimeError(
            f"Only found {len(selected)} valid eval samples, requested {num_samples}. "
            f"First missing entries: {missing[:5]}"
        )
    return selected


def summarize_checkpoint_rows(rows: list[dict[str, Any]], failures: list[dict[str, Any]], requested: int) -> dict[str, Any]:
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


def process_single_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    rows: list[dict[str, str]],
    data_dir: Path,
    latent_model: str,
    output_dir: Path,
    point_samples: int,
    seed: int,
    device: Any,
    skip_existing_meshes: bool = False,
    require_all_samples: bool = False,
) -> dict[str, Any]:
    import torch

    from eval.common.mesh_metrics import compare_meshes, load_mesh
    from eval.common.model_loading import (
        build_stable3dgen_mesh_decoder,
        decode_latent_to_mesh_result,
        export_stable3dgen_mesh,
        load_decoder_checkpoint,
    )

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
        lp = latent_path(data_dir, latent_model, sha)
        gt_mesh_path = render_mesh_path(data_dir, sha)
        pred_mesh_path = mesh_dir / f"{sha}.ply"
        try:
            if skip_existing_meshes and pred_mesh_path.is_file():
                pred_mesh = load_mesh(pred_mesh_path)
            else:
                mesh_result = decode_latent_to_mesh_result(decoder, lp, device)
                pred_mesh = export_stable3dgen_mesh(mesh_result, pred_mesh_path)
            metrics = compare_meshes(
                pred_mesh,
                load_mesh(gt_mesh_path),
                point_samples=point_samples,
                seed=seed + sample_index * 1009,
            )
            metrics.update({
                "checkpoint": tag,
                "checkpoint_path": str(checkpoint),
                "sample_index": sample_index,
                "sha256": sha,
                "pred_mesh_path": str(pred_mesh_path),
                "gt_mesh_path": str(gt_mesh_path),
            })
            per_sample.append(metrics)
            print(f"[{sample_index + 1}/{len(rows)}] OK {sha}", flush=True)
        except Exception as exc:
            failure = {
                "checkpoint": tag,
                "sample_index": sample_index,
                "sha256": sha,
                "error": repr(exc),
                "latent_path": str(lp),
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
    write_json(failures_dir / f"{tag}_failures.json", failures)
    summary = summarize_checkpoint_rows(per_sample, failures, len(rows))
    summary.update({"checkpoint": tag, "checkpoint_path": str(checkpoint)})
    write_json(metrics_dir / f"{tag}_summary.json", summary)
    del decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def process_many_checkpoints(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    config = load_json(args.config)
    metadata_path = args.metadata or args.data_dir / "metadata.csv"
    rows = select_eval_rows(args.data_dir, read_metadata_rows(metadata_path), args.latent_model, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_samples.json", [row["sha256"] for row in rows])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    summaries = [
        process_single_checkpoint(
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
        for checkpoint in args.checkpoints
    ]
    write_csv(args.output_dir / "metrics" / "all_checkpoints_summary.csv", summaries)
    return summaries


def process_summary_compare(args: argparse.Namespace) -> list[dict[str, Any]]:
    return merge_summary_files(
        args.summaries,
        args.output_csv,
        sort_by=args.sort_by,
        descending=args.descending,
    )


def add_eval_args(parser: argparse.ArgumentParser) -> None:
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0].startswith("-"):
        parser = argparse.ArgumentParser(description=__doc__)
        add_eval_args(parser)
        args = parser.parse_args(argv)
        args.command = "many"
        return args

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single", help="Evaluate one checkpoint.")
    add_eval_args(single)
    many = subparsers.add_parser("many", help="Evaluate one or more checkpoints.")
    add_eval_args(many)
    compare = subparsers.add_parser("compare", help="Merge and sort checkpoint summary JSON files.")
    compare.add_argument("summaries", nargs="+", type=Path, help="*_summary.json files")
    compare.add_argument("--output_csv", type=Path, required=True)
    compare.add_argument("--sort_by", default="chamfer_l1_mean")
    compare.add_argument("--descending", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "single" and len(args.checkpoints) != 1:
        raise ValueError("single mode requires exactly one --checkpoints entry")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "compare":
        rows = process_summary_compare(args)
        print(f"Wrote {len(rows)} summary rows to {args.output_csv}")
        return
    summaries = process_many_checkpoints(args)
    print(json.dumps({"checkpoints": summaries, "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
