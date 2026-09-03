"""SLat Flow 生成产物的公共读取与参数解析工具。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from eval.common.io import load_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_flow_manifest(run_dir: Path) -> list[dict[str, str]]:
    """读取一次 SLat Flow 生成任务的样本清单。"""
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Flow manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Flow manifest is empty: {manifest_path}")
    return rows


def resolve_repo_path(path: Path) -> Path:
    """将生成清单中相对仓库根目录记录的路径转成绝对路径。"""
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_flow_config(run_dir: Path) -> dict[str, Any]:
    """从生成任务 summary 中定位并读取训练配置。"""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Flow summary not found: {summary_path}")
    summary = load_json(summary_path)
    config_path = resolve_repo_path(Path(summary["config"]))
    if not config_path.is_file():
        raise FileNotFoundError(f"Flow config recorded in summary not found: {config_path}")
    return load_json(config_path)


def latent_normalization_from_config(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 SLat dataset 的逐通道均值和标准差。"""
    normalization = config.get("dataset", {}).get("args", {}).get("normalization")
    if normalization is None:
        return None
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, -1)
    return mean, std


def resolve_generated_latent_path(
    run_dir: Path,
    manifest_row: dict[str, str],
) -> Path:
    """定位某个生成样本的 generated_latent.npz。"""
    latent_text = manifest_row.get("generated_latent_path", "").strip()
    if latent_text:
        return resolve_repo_path(Path(latent_text))
    return run_dir / "samples" / manifest_row["sample_id"] / "generated_latent.npz"


def load_generated_latent(
    latent_path: Path,
    *,
    normalization: tuple[np.ndarray, np.ndarray] | None,
    denormalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """读取生成 latent，并按 flow 配置选择是否反归一化。"""
    if not latent_path.is_file():
        raise FileNotFoundError(f"Generated latent not found: {latent_path}")
    with np.load(latent_path, allow_pickle=False) as data:
        for key in ("coords", "feats"):
            if key not in data.files:
                raise KeyError(f"{latent_path} missing key {key!r}; keys={data.files}")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{latent_path}: expected coords shape (N, 3), got {coords.shape}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(f"{latent_path}: coords/feats mismatch: {coords.shape} vs {feats.shape}")
    if not np.isfinite(feats).all():
        raise ValueError(f"{latent_path}: latent features contain non-finite values")
    if denormalize and normalization is not None:
        mean, std = normalization
        if feats.shape[1] != mean.shape[1]:
            raise ValueError(
                f"{latent_path}: feats channels {feats.shape[1]} "
                f"!= normalization channels {mean.shape[1]}"
            )
        feats = feats * std + mean
    return coords, feats


def parse_mesh_decoder_specs(
    specs: list[str] | None,
) -> dict[str, tuple[Path, Path]]:
    """解析 NAME=MESH_CONFIG=MESH_DECODER_CKPT 参数。"""
    if not specs:
        return {}
    parsed: dict[str, tuple[Path, Path]] = {}
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                "Mesh decoder spec must be "
                f"NAME=MESH_CONFIG=MESH_DECODER_CKPT, got: {spec}"
            )
        name, config_path, checkpoint_path = parts
        name = name.strip()
        if not name:
            raise ValueError(f"Empty run name in mesh decoder spec: {spec}")
        if name in parsed:
            raise ValueError(f"Duplicate mesh decoder spec for run: {name}")
        parsed[name] = (Path(config_path), Path(checkpoint_path))
    return parsed
