"""Small IO helpers shared by evaluation scripts."""

# 中文说明：JSON/CSV 写入、目录创建、安全 tag 和 NAME=PATH 参数解析等通用 IO 工具。

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_tag(text: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return tag or "run"


def parse_name_path_specs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Run name is empty in spec: {spec}")
        if name in runs:
            raise ValueError(f"Duplicate run name: {name}")
        runs[name] = Path(path)
    if not runs:
        raise ValueError("At least one run spec is required.")
    return runs
