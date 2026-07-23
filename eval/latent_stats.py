#!/usr/bin/env python3
"""Analyze SLat latent feature distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import summarize_numeric_values


def read_metadata_latent_files(data_dir: Path, latent_model: str, limit: int | None = None) -> list[Path]:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    latent_flag = f"latent_{latent_model}"
    paths: list[Path] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "sha256" not in (reader.fieldnames or []):
            raise KeyError(f"metadata.csv must contain sha256 column: {metadata_path}")
        for row in reader:
            if latent_flag in row and str(row[latent_flag]).strip().lower() not in {"1", "true", "yes", "y"}:
                continue
            paths.append(data_dir / "latents" / latent_model / f"{row['sha256']}.npz")
            if limit is not None and len(paths) >= limit:
                break
    return paths


def summarize_latent_files(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    all_finite_values: list[np.ndarray] = []
    finite_count = 0
    total_count = 0

    for path in paths:
        row: dict[str, Any] = {"path": str(path), "sha256": path.stem}
        try:
            with np.load(path, allow_pickle=False) as data:
                if "feats" not in data.files or "coords" not in data.files:
                    raise KeyError(f"{path} missing feats or coords; keys={data.files}")
                feats = np.asarray(data["feats"], dtype=np.float64)
                coords = np.asarray(data["coords"])
            if feats.ndim != 2:
                raise ValueError(f"{path}: expected feats shape (N, C), got {feats.shape}")
            if coords.ndim != 2 or coords.shape[0] != feats.shape[0]:
                raise ValueError(f"{path}: coords/feats mismatch: {coords.shape} vs {feats.shape}")

            finite_mask = np.isfinite(feats)
            finite_vals = feats[finite_mask]
            finite_count += int(finite_mask.sum())
            total_count += int(feats.size)
            if finite_vals.size:
                all_finite_values.append(finite_vals)
            row.update({
                "token_count": int(feats.shape[0]),
                "channels": int(feats.shape[1]),
                "finite_rate": float(finite_mask.mean()) if feats.size else math.nan,
                "mean": float(finite_vals.mean()) if finite_vals.size else math.nan,
                "std": float(finite_vals.std()) if finite_vals.size else math.nan,
                "min": float(finite_vals.min()) if finite_vals.size else math.nan,
                "max": float(finite_vals.max()) if finite_vals.size else math.nan,
                "norm_mean": float(np.linalg.norm(feats, axis=1).mean()) if feats.shape[0] else math.nan,
                "coord_min": int(coords.min()) if coords.size else math.nan,
                "coord_max": int(coords.max()) if coords.size else math.nan,
                "failed": False,
                "error": "",
            })
        except Exception as exc:
            row.update({"failed": True, "error": repr(exc)})
            failed.append({"path": str(path), "error": repr(exc)})
        rows.append(row)

    merged = np.concatenate(all_finite_values) if all_finite_values else np.array([], dtype=np.float64)
    summary = {
        "num_files": len(paths),
        "num_records": len(rows),
        "failed_count": len(failed),
        "token_count": summarize_numeric_values([float(row["token_count"]) for row in rows if not row.get("failed")]),
        "per_sample_norm_mean": summarize_numeric_values([float(row["norm_mean"]) for row in rows if not row.get("failed")]),
        "feats": {
            **summarize_numeric_values([float(v) for v in merged]),
            "finite_count": int(finite_count),
            "total_count": int(total_count),
            "finite_rate": float(finite_count / total_count) if total_count else math.nan,
        },
        "failed": failed,
    }
    if "sum" not in summary["token_count"]:
        summary["token_count"]["sum"] = int(sum(row["token_count"] for row in rows if not row.get("failed")))
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = read_metadata_latent_files(args.data_dir, args.latent_model, args.num_samples)
    summary, rows = summarize_latent_files(paths)
    summary.update({
        "data_dir": str(args.data_dir),
        "latent_model": args.latent_model,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--latent_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
