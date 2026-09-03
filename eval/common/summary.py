"""Summary helpers shared by evaluation scripts."""

# 中文说明：数值指标汇总、summary 展平和多 run comparison CSV 生成工具。

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.common.io import load_json, write_csv


def summarize_numeric_values(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {}
    arr = np.array(finite, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def summarize_metric_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] != ""]
        stats = summarize_numeric_values(values)
        if stats:
            summary[metric] = stats
    return summary


def flatten_summary_metrics(name: str, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
        "num_records": summary.get("num_records", summary.get("successful_samples", 0)),
        "failed_count": summary.get("failed_count", summary.get("failed_samples", 0)),
    }
    metrics = summary.get("metrics", {})
    if isinstance(metrics, dict):
        for metric_name, metric_stats in metrics.items():
            if isinstance(metric_stats, dict):
                for stat_name, value in metric_stats.items():
                    row[f"{stat_name}_{metric_name}"] = value
    for key, value in summary.items():
        if key.endswith(("_mean", "_median", "_std")):
            row[key] = value
    return row


def compare_summary_dirs(runs: dict[str, Path], output_csv: Path) -> list[dict[str, Any]]:
    rows = [flatten_summary_metrics(name, run_dir, load_json(run_dir / "summary.json")) for name, run_dir in runs.items()]
    leading = ["name", "run_dir", "num_records", "failed_count"]
    fieldnames = leading + sorted({key for row in rows for key in row} - set(leading))
    write_csv(output_csv, rows, fieldnames)
    return rows


def merge_summary_files(
    summaries: list[Path],
    output_csv: Path,
    *,
    sort_by: str = "chamfer_l1_mean",
    descending: bool = False,
) -> list[dict[str, Any]]:
    rows = [load_json(path) for path in summaries]
    rows.sort(key=lambda row: float(row.get(sort_by, "inf")), reverse=descending)
    write_csv(output_csv, rows)
    return rows
