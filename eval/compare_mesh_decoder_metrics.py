#!/usr/bin/env python3
"""Compare mesh decoder eval summary files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and sort mesh decoder eval summaries.")
    parser.add_argument("summaries", nargs="+", type=Path, help="*_summary.json files")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--sort_by", default="chamfer_l1_mean")
    parser.add_argument("--descending", action="store_true")
    return parser.parse_args(argv)


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


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    rows.sort(
        key=lambda row: float(row.get(args.sort_by, "inf")),
        reverse=args.descending,
    )
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} summary rows to {args.output_csv}")


if __name__ == "__main__":
    main()

