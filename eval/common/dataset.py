"""Dataset file discovery helpers for evaluation scripts."""

# 中文说明：数据集文件发现工具，负责读取 metadata、解析固定视角/索引，以及拼接 latent/mesh 路径。

from __future__ import annotations

import csv
import random
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}


def read_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {metadata_path}")
    if "sha256" not in rows[0]:
        raise KeyError(f"metadata.csv must contain sha256 column: {metadata_path}")
    return rows


def parse_view_indices(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("view_indices must contain at least one integer")
    return values


def parse_indices(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def select_indices(length: int, num_samples: int, seed: int, indices: list[int] | None = None) -> list[int]:
    if indices is None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        selected = random.Random(seed).sample(range(length), min(num_samples, length))
    else:
        selected = indices
    for index in selected:
        if index < 0 or index >= length:
            raise IndexError(f"Dataset index {index} out of range for len={length}")
    return selected


def latent_path(data_dir: Path, latent_model: str, sha: str) -> Path:
    return data_dir / "latents" / latent_model / f"{sha}.npz"


def render_mesh_path(data_dir: Path, sha: str) -> Path:
    return data_dir / "renders" / sha / "mesh.ply"
