#!/usr/bin/env python3
"""Temporary helper: dedupe trainer log lines by numeric step, keeping the latest line."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(r"^(\d+):\s")


def dedupe_step_log(path: Path, *, dry_run: bool = False) -> tuple[Path | None, int, int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    latest_by_step: OrderedDict[int, str] = OrderedDict()
    passthrough: list[str] = []

    for line in lines:
        match = STEP_RE.match(line)
        if match is None:
            passthrough.append(line)
            continue
        step = int(match.group(1))
        latest_by_step[step] = line

    deduped = [latest_by_step[step] for step in sorted(latest_by_step)]
    deduped.extend(passthrough)
    duplicate_count = len(lines) - len(deduped)

    backup_path = None
    if not dry_run and duplicate_count > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
        shutil.copy2(path, backup_path)
        path.write_text("\n".join(deduped) + "\n", encoding="utf-8")

    return backup_path, len(lines), len(deduped), duplicate_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_path, before, after, duplicates = dedupe_step_log(args.log_path, dry_run=args.dry_run)
    print(f"log_path={args.log_path}")
    print(f"before_lines={before}")
    print(f"after_lines={after}")
    print(f"duplicates_removed={duplicates}")
    if backup_path is not None:
        print(f"backup_path={backup_path}")
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
