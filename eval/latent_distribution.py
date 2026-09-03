#!/usr/bin/env python3
"""Evaluate SLat latent feature distributions."""

# 中文说明：
# 评估已经编码好的 SLat latent 分布，不做模型前向。
# 输入是数据集 metadata.csv 和 latents/<latent_model>/<sha>.npz。
# 输出是逐样本统计 per_sample.csv 和整体汇总 summary.json。

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.dataset import TRUE_VALUES, latent_path, read_metadata_rows
from eval.common.io import write_csv, write_json
from eval.common.summary import summarize_numeric_values


def collect_latent_files(data_dir: Path, latent_model: str, limit: int | None = None) -> list[Path]:
    rows = read_metadata_rows(data_dir / "metadata.csv")
    latent_flag = f"latent_{latent_model}"
    paths: list[Path] = []
    for row in rows:
        if latent_flag in row and str(row[latent_flag]).strip().lower() not in TRUE_VALUES:
            continue
        paths.append(latent_path(data_dir, latent_model, row["sha256"]))
        if limit is not None and len(paths) >= limit:
            break
    return paths


def compute_single_latent_stats(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    row: dict[str, Any] = {"path": str(path), "sha256": path.stem}
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
    return row, finite_vals


def summarize_latent_rows(
    paths: list[Path],
    rows: list[dict[str, Any]],
    failed: list[dict[str, str]],
    finite_values: list[np.ndarray],
) -> dict[str, Any]:
    merged = np.concatenate(finite_values) if finite_values else np.array([], dtype=np.float64)
    finite_count = int(sum(values.size for values in finite_values))
    total_count = int(sum(int(row.get("token_count", 0)) * int(row.get("channels", 0)) for row in rows if not row.get("failed")))
    token_summary = summarize_numeric_values([float(row["token_count"]) for row in rows if not row.get("failed")])
    token_summary["sum"] = int(sum(row["token_count"] for row in rows if not row.get("failed")))
    return {
        "num_files": len(paths),
        "num_records": len(rows),
        "failed_count": len(failed),
        "token_count": token_summary,
        "per_sample_norm_mean": summarize_numeric_values([float(row["norm_mean"]) for row in rows if not row.get("failed")]),
        "feats": {
            **summarize_numeric_values([float(v) for v in merged]),
            "finite_count": finite_count,
            "total_count": total_count,
            "finite_rate": float(finite_count / total_count) if total_count else math.nan,
        },
        "failed": failed,
    }


def process_many(data_dir: Path, latent_model: str, output_dir: Path, num_samples: int | None = None) -> dict[str, Any]:
    paths = collect_latent_files(data_dir, latent_model, num_samples)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    finite_values: list[np.ndarray] = []
    for path in paths:
        try:
            row, finite = compute_single_latent_stats(path)
            finite_values.append(finite)
        except Exception as exc:
            row = {"path": str(path), "sha256": path.stem, "failed": True, "error": repr(exc)}
            failed.append({"path": str(path), "error": repr(exc)})
        rows.append(row)
    summary = summarize_latent_rows(paths, rows, failed, finite_values)
    summary.update({"data_dir": str(data_dir), "latent_model": latent_model})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--latent_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(process_many(args.data_dir, args.latent_model, args.output_dir, args.num_samples), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
