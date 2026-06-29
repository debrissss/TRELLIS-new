#!/usr/bin/env python3
"""Remove duplicate step entries from TRELLIS line-oriented training logs.

The trainer log format is expected to be one record per line:

    123: {"time": ...}

When a step appears multiple times, this script keeps the last occurrence.
Output is sorted by numeric step so downstream plotting sees a monotonic log.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate TRELLIS training log steps, keeping the newest record."
    )
    parser.add_argument("log_path", type=Path, help="Path to log.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <log_path>.dedup.txt unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input log after writing a .bak backup.",
    )
    return parser.parse_args()


def parse_step(line: str, line_number: int) -> int:
    prefix, sep, _ = line.partition(":")
    if not sep:
        raise ValueError(f"line {line_number}: missing ':' step separator")
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid step {prefix!r}") from exc


def deduplicate(log_path: Path) -> tuple[dict[int, str], int, int]:
    latest_by_step: dict[int, str] = {}
    total_lines = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_lines += 1
            latest_by_step[parse_step(line, line_number)] = line

    duplicates_removed = total_lines - len(latest_by_step)
    return latest_by_step, total_lines, duplicates_removed


def default_output_path(log_path: Path) -> Path:
    if log_path.suffix:
        return log_path.with_name(f"{log_path.stem}.dedup{log_path.suffix}")
    return log_path.with_name(f"{log_path.name}.dedup")


def main() -> None:
    args = parse_args()
    log_path = args.log_path
    if not log_path.is_file():
        raise SystemExit(f"log file not found: {log_path}")
    if args.in_place and args.output:
        raise SystemExit("--output cannot be used with --in-place")

    latest_by_step, total_lines, duplicates_removed = deduplicate(log_path)

    if args.in_place:
        output_path = log_path
        backup_path = log_path.with_name(f"{log_path.name}.bak")
        backup_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        output_path = args.output or default_output_path(log_path)

    with output_path.open("w", encoding="utf-8") as handle:
        for step in sorted(latest_by_step):
            handle.write(latest_by_step[step])

    print(f"input_lines={total_lines}")
    print(f"unique_steps={len(latest_by_step)}")
    print(f"duplicates_removed={duplicates_removed}")
    print(f"output={output_path}")
    if args.in_place:
        print(f"backup={backup_path}")


if __name__ == "__main__":
    main()
