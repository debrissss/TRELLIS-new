#!/usr/bin/env python3
"""Compare fixed SLat flow generation outputs against GT render grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    l1_metric,
    lpips_metric,
    metric_value,
    mse_metric,
    psnr_metric,
    ssim_metric,
    summarize_metric_rows,
)


METRIC_NAMES = ["l1", "mse", "psnr", "ssim", "ssim_loss", "lpips", "mask_iou"]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def foreground_mask(image: Image.Image, threshold: int = 8) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(arr > threshold, axis=-1)


def mask_iou(pred: Image.Image, target: Image.Image) -> float:
    pred_mask = foreground_mask(pred)
    target_mask = foreground_mask(target)
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_mask, target_mask).sum()
    return float(intersection / union)


def compute_image_pair_metrics(pred: Image.Image, target: Image.Image, *, skip_lpips: bool = False) -> dict[str, Any]:
    if pred.size != target.size:
        pred = pred.resize(target.size, Image.Resampling.LANCZOS)
    pred_t = image_to_tensor(pred)
    target_t = image_to_tensor(target)
    ssim = ssim_metric(pred_t, target_t)
    row: dict[str, Any] = {
        "failed": False,
        "error": "",
        "l1": metric_value(l1_metric(pred_t, target_t)),
        "mse": metric_value(mse_metric(pred_t, target_t)),
        "psnr": metric_value(psnr_metric(pred_t, target_t)),
        "ssim": metric_value(ssim),
        "ssim_loss": metric_value(1 - ssim),
        "mask_iou": mask_iou(pred, target),
    }
    if skip_lpips:
        row["lpips"] = math.nan
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        row["lpips"] = metric_value(lpips_metric(pred_t.to(device), target_t.to(device)))
    return row


def iter_sample_pairs(run_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    samples_root = run_dir / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"samples directory not found: {samples_root}")
    for sample_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        for generated in sorted(sample_dir.glob("generated_*.png")):
            suffix = generated.name.replace("generated_", "")
            target = sample_dir / f"gt_{suffix}"
            if target.is_file():
                pairs.append((sample_dir.name, generated, target))
    if not pairs:
        raise FileNotFoundError(f"No generated_*.png / gt_*.png pairs found in {samples_root}")
    return pairs


def evaluate_run(name: str, run_dir: Path, *, skip_lpips: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failed = []
    for sample_id, generated_path, target_path in iter_sample_pairs(run_dir):
        row: dict[str, Any] = {
            "run": name,
            "sample_id": sample_id,
            "generated_path": str(generated_path),
            "gt_path": str(target_path),
        }
        try:
            metrics = compute_image_pair_metrics(Image.open(generated_path), Image.open(target_path), skip_lpips=skip_lpips)
            row.update(metrics)
        except Exception as exc:
            row.update({"failed": True, "error": repr(exc)})
            failed.append(row)
        rows.append(row)
    ok_rows = [row for row in rows if not row.get("failed")]
    summary = {
        "run_dir": str(run_dir),
        "num_records": len(rows),
        "failed_count": len(failed),
        "metrics": summarize_metric_rows(ok_rows, METRIC_NAMES),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "sample_id",
        "generated_path",
        "gt_path",
        "failed",
        "error",
        *METRIC_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compare_generation_runs(runs: dict[str, Path], output_dir: Path, *, skip_lpips: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries = {}
    for name, run_dir in runs.items():
        summary, rows = evaluate_run(name, run_dir, skip_lpips=skip_lpips)
        summaries[name] = summary
        all_rows.extend(rows)

    failed_count = sum(1 for row in all_rows if row.get("failed"))
    comparison = {
        "runs": summaries,
        "num_records": len(all_rows),
        "failed_count": failed_count,
        "skip_lpips": bool(skip_lpips),
    }
    write_csv(output_dir / "comparison.csv", all_rows)
    (output_dir / "summary.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison


def parse_run_specs(specs: list[str]) -> dict[str, Path]:
    runs = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be NAME=DIR, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Run name is empty in spec: {spec}")
        if name in runs:
            raise ValueError(f"Duplicate run name: {name}")
        runs[name] = Path(path)
    if not runs:
        raise ValueError("At least one --runs NAME=DIR entry is required")
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One or more NAME=flow_generation_dir entries.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_generation_runs(parse_run_specs(args.runs), args.output_dir, skip_lpips=args.skip_lpips)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
