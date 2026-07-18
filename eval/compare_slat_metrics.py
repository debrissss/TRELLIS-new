#!/usr/bin/env python3
"""Compare SLat encoder/decoder reconstruction evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ["loss", "rec", "l1", "mse", "psnr", "ssim_loss", "lpips", "kl"]
DEFAULT_STATS = ["mean", "median", "p50", "p90", "min", "max"]


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_comparison_rows(runs: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for name, run_dir in runs.items():
        summary = load_summary(run_dir)
        row: dict[str, Any] = {
            "name": name,
            "run_dir": str(run_dir),
            "num_records": summary.get("num_records", 0),
            "failed_count": summary.get("failed_count", 0),
        }
        metrics = summary.get("metrics", {})
        for metric_name, metric_stats in metrics.items():
            if not isinstance(metric_stats, dict):
                continue
            for stat_name, value in metric_stats.items():
                row[f"{stat_name}_{metric_name}"] = value
        rows.append(row)
    return rows


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    leading = ["name", "run_dir", "num_records", "failed_count"]
    preferred = [f"{stat}_{metric}" for metric in DEFAULT_METRICS for stat in DEFAULT_STATS]
    remaining = sorted({key for row in rows for key in row} - set(leading) - set(preferred))
    return leading + [key for key in preferred if any(key in row for row in rows)] + remaining


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
        raise ValueError("At least one --runs NAME=DIR entry is required.")
    return runs


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One or more NAME=eval_output_dir entries.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_comparison_rows(parse_run_specs(args.runs))
    write_csv(rows, args.output)
    print(f"[OK] wrote comparison CSV: {args.output}")


if __name__ == "__main__":
    main()
