#!/usr/bin/env python3
"""Prepare a fixed FaceScape evaluation subset for checkpoint comparison."""

# 中文说明：
# 从 FaceScape split 中抽取固定评估子集，保证不同 checkpoint 使用相同样本。
# 输出 metadata.csv、selected_sha256.txt、manifest.json，并软链接或复制 renders/features。

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PrepareResult:
    selected_count: int
    output_dir: Path
    selected_sha256: list[str]


def parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_metadata(metadata_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"metadata.csv has no header: {metadata_path}")
    if "sha256" not in fieldnames:
        raise KeyError(f"metadata.csv must contain sha256 column: {metadata_path}")
    return rows, fieldnames


def feature_npz_is_readable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return "indices" in data.files and "patchtokens" in data.files
    except Exception:
        return False


def valid_eval_rows(
    rows: list[dict[str, str]],
    source_dir: Path,
    feature_model: str,
) -> list[dict[str, str]]:
    feature_flag = f"feature_{feature_model}"
    valid = []
    for row in rows:
        sha = row.get("sha256", "")
        if not sha:
            continue
        if feature_flag in row and not parse_boolish(row[feature_flag]):
            continue
        render_dir = source_dir / "renders" / sha
        transforms_path = render_dir / "transforms.json"
        feature_path = source_dir / "features" / feature_model / f"{sha}.npz"
        if not transforms_path.is_file():
            continue
        if not feature_npz_is_readable(feature_path):
            continue
        valid.append(row)
    return valid


def select_rows(rows: list[dict[str, str]], num_samples: int, seed: int) -> list[dict[str, str]]:
    if num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if len(rows) < num_samples:
        raise ValueError(f"Only {len(rows)} valid samples available; requested {num_samples}.")
    rng = random.Random(seed)
    selected_shas = rng.sample(sorted(row["sha256"] for row in rows), num_samples)
    by_sha = {row["sha256"]: row for row in rows}
    return [by_sha[sha] for sha in selected_shas]


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def write_metadata(output_dir: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_eval_subset(
    *,
    source_dir: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    feature_model: str,
    copy_files: bool,
    overwrite: bool,
) -> PrepareResult:
    source_dir = source_dir.resolve()
    rows, fieldnames = read_metadata(source_dir / "metadata.csv")
    selected = select_rows(valid_eval_rows(rows, source_dir, feature_model), num_samples, seed)

    reset_output_dir(output_dir, overwrite=overwrite)
    write_metadata(output_dir, selected, fieldnames)
    selected_shas = [row["sha256"] for row in selected]
    (output_dir / "selected_sha256.txt").write_text(
        "\n".join(selected_shas) + "\n",
        encoding="utf-8",
    )

    for sha in selected_shas:
        link_or_copy(source_dir / "renders" / sha, output_dir / "renders" / sha, copy_files)
        link_or_copy(
            source_dir / "features" / feature_model / f"{sha}.npz",
            output_dir / "features" / feature_model / f"{sha}.npz",
            copy_files,
        )

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "feature_model": feature_model,
        "storage": "copy" if copy_files else "symlink",
        "metadata_rows": len(selected),
        "required_layout": {
            "metadata": "metadata.csv",
            "renders": "renders/<sha>/transforms.json and image files",
            "features": f"features/{feature_model}/<sha>.npz",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return PrepareResult(selected_count=len(selected), output_dir=output_dir, selected_sha256=selected_shas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--feature_model", default="dinov2_vitl14_reg")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_eval_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        feature_model=args.feature_model,
        copy_files=args.copy,
        overwrite=args.overwrite,
    )
    print(f"[OK] wrote {result.selected_count} eval samples to {result.output_dir}")


if __name__ == "__main__":
    main()
