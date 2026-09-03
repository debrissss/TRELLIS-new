"""Shared artifact and manifest helpers for split SLat inference.

The model forward passes remain owned by the current TRELLIS and Stable3DGen
implementations. This module only defines the file boundary shared by the
independent SLat encoder, flow, and decoder processes.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval.common.dataset import TRUE_VALUES, read_metadata_rows
from eval.common.io import write_csv
from eval.common.ss_inference import (
    list_condition_images,
    parse_indices,
    read_stage_manifest,
)


SLAT_INPUT_MANIFEST_FIELDS = [
    "sample_id",
    "dataset_index",
    "data_dir",
    "feature_model",
    "feature_path",
    "condition_image_path",
]
SLAT_COORDS_KEY = "coords"
SLAT_FEATS_KEY = "feats"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _select_indices(
    length: int,
    *,
    num_samples: int,
    seed: int,
    indices: list[int] | None,
) -> list[int]:
    if indices is not None:
        selected = indices
    elif num_samples <= 0 or num_samples >= length:
        selected = list(range(length))
    else:
        selected = random.Random(seed).sample(range(length), num_samples)
    for index in selected:
        if index < 0 or index >= length:
            raise IndexError(f"Input index {index} out of range for {length} valid samples")
    return selected


def _infer_data_dir_from_feature_path(feature_path: Path, feature_model: str) -> Path:
    expected_parent = Path("features") / feature_model
    if feature_path.parent.name != expected_parent.name:
        raise ValueError(
            "Cannot infer data_dir from feature_path; provide a data_dir column in the manifest: "
            f"{feature_path}"
        )
    features_dir = feature_path.parent.parent
    if features_dir.name != "features":
        raise ValueError(
            "Expected feature_path under data_dir/features/<feature_model>/: "
            f"{feature_path}"
        )
    return features_dir.parent.resolve()


def select_slat_encoder_inputs(
    data_dir: Path,
    *,
    feature_model: str,
    num_samples: int,
    seed: int,
    indices: list[int] | None,
    min_aesthetic_score: float | None,
    max_num_voxels: int | None,
) -> list[dict[str, Any]]:
    """Select feature-ready samples using the current SparseFeat dataset rules."""

    data_dir = _absolute(data_dir)
    metadata_rows = read_metadata_rows(data_dir / "metadata.csv")
    feature_column = f"feature_{feature_model}"
    candidates: list[dict[str, Any]] = []
    for metadata_index, row in enumerate(metadata_rows):
        sample_id = row["sha256"].strip()
        if not sample_id:
            continue
        if feature_column in row and not _truthy(row[feature_column]):
            continue
        if (
            min_aesthetic_score is not None
            and row.get("aesthetic_score", "").strip()
            and float(row["aesthetic_score"]) < min_aesthetic_score
        ):
            continue
        if (
            max_num_voxels is not None
            and row.get("num_voxels", "").strip()
            and int(float(row["num_voxels"])) > max_num_voxels
        ):
            continue

        feature_path = (
            data_dir / "features" / feature_model / f"{sample_id}.npz"
        ).resolve()
        if not feature_path.is_file():
            continue
        condition_images = list_condition_images(data_dir, sample_id)
        candidates.append(
            {
                "sample_id": sample_id,
                "dataset_index": metadata_index,
                "data_dir": str(data_dir),
                "feature_model": feature_model,
                "feature_path": str(feature_path),
                "_condition_images": condition_images,
            }
        )

    if not candidates:
        raise ValueError(
            f"No feature-ready SLat encoder samples for model {feature_model!r} under {data_dir}"
        )

    selected_indices = _select_indices(
        len(candidates),
        num_samples=num_samples,
        seed=seed,
        indices=indices,
    )
    condition_rng = random.Random(seed + 1_000_003)
    selected: list[dict[str, Any]] = []
    for candidate_index in selected_indices:
        candidate = dict(candidates[candidate_index])
        condition_images = candidate.pop("_condition_images")
        condition_path = condition_rng.choice(condition_images) if condition_images else None
        candidate["condition_image_path"] = str(condition_path) if condition_path else ""
        selected.append(candidate)
    return selected


def read_slat_encoder_input_manifest(
    manifest_path: Path,
    *,
    default_feature_model: str,
) -> list[dict[str, Any]]:
    manifest_path = _absolute(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        raw_rows = list(csv.DictReader(file))
    if not raw_rows:
        raise ValueError(f"Input manifest is empty: {manifest_path}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in enumerate(raw_rows):
        sample_id = (raw.get("sample_id") or raw.get("sha256") or "").strip()
        if not sample_id:
            raise KeyError(f"Row {row_index} has no sample_id/sha256: {manifest_path}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {manifest_path}")
        seen.add(sample_id)

        feature_model = (raw.get("feature_model") or default_feature_model).strip()
        feature_text = (raw.get("feature_path") or "").strip()
        if not feature_text:
            raise KeyError(f"{sample_id}: feature_path is required in {manifest_path}")
        feature_path = _resolve_manifest_path(feature_text, manifest_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"{sample_id}: feature artifact not found: {feature_path}")

        data_dir_text = (raw.get("data_dir") or "").strip()
        data_dir = (
            _resolve_manifest_path(data_dir_text, manifest_path)
            if data_dir_text
            else _infer_data_dir_from_feature_path(feature_path, feature_model)
        )
        expected_feature_path = (
            data_dir / "features" / feature_model / f"{sample_id}.npz"
        ).resolve()
        if feature_path != expected_feature_path:
            raise ValueError(
                f"{sample_id}: feature_path must match the current dataset layout "
                f"{expected_feature_path}, got {feature_path}"
            )
        condition_text = (raw.get("condition_image_path") or "").strip()
        condition_path = (
            _resolve_manifest_path(condition_text, manifest_path)
            if condition_text
            else None
        )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": int(raw.get("dataset_index") or row_index),
                "data_dir": str(data_dir),
                "feature_model": feature_model,
                "feature_path": str(feature_path),
                "condition_image_path": str(condition_path) if condition_path else "",
            }
        )
    return rows


def resolve_slat_encoder_inputs(
    *,
    data_dir: Path | None,
    input_manifest: Path | None,
    feature_model: str,
    num_samples: int,
    seed: int,
    indices: str | None,
    min_aesthetic_score: float | None,
    max_num_voxels: int | None,
) -> list[dict[str, Any]]:
    if (data_dir is None) == (input_manifest is None):
        raise ValueError("Exactly one of --data_dir or --input_manifest must be provided")
    if input_manifest is not None:
        return read_slat_encoder_input_manifest(
            input_manifest,
            default_feature_model=feature_model,
        )
    assert data_dir is not None
    return select_slat_encoder_inputs(
        data_dir,
        feature_model=feature_model,
        num_samples=num_samples,
        seed=seed,
        indices=parse_indices(indices),
        min_aesthetic_score=min_aesthetic_score,
        max_num_voxels=max_num_voxels,
    )


def write_slat_input_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, fieldnames=SLAT_INPUT_MANIFEST_FIELDS)


def successful_ss_coords_rows(
    manifest_path: Path,
    *,
    num_samples: int,
    seed: int,
    indices: str | None,
) -> list[dict[str, Any]]:
    """Read successful SS decoder rows for the SLat flow handoff."""

    manifest_path = _absolute(manifest_path)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in enumerate(read_stage_manifest(manifest_path)):
        if _truthy(raw.get("failed", "false")):
            continue
        sample_id = (raw.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {manifest_path}")
        seen.add(sample_id)

        coords_text = (raw.get("coords_path") or "").strip()
        if not coords_text:
            continue
        coords_path = _resolve_manifest_path(coords_text, manifest_path)
        if not coords_path.is_file():
            raise FileNotFoundError(f"{sample_id}: SS coords artifact not found: {coords_path}")

        original_text = (raw.get("condition_image_path") or "").strip()
        prepared_text = (raw.get("prepared_condition_path") or "").strip()
        condition_features_text = (
            raw.get("condition_features_path") or ""
        ).strip()
        rng_state_text = (raw.get("rng_state_path") or "").strip()
        original_path = (
            _resolve_manifest_path(original_text, manifest_path) if original_text else None
        )
        prepared_path = (
            _resolve_manifest_path(prepared_text, manifest_path) if prepared_text else None
        )
        condition_features_path = (
            _resolve_manifest_path(condition_features_text, manifest_path)
            if condition_features_text
            else None
        )
        if (
            condition_features_path is not None
            and not condition_features_path.is_file()
        ):
            raise FileNotFoundError(
                f"{sample_id}: image condition artifact not found: "
                f"{condition_features_path}"
            )
        rng_state_path = (
            _resolve_manifest_path(rng_state_text, manifest_path)
            if rng_state_text
            else None
        )
        if rng_state_path is not None and not rng_state_path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: SS RNG state artifact not found: {rng_state_path}"
            )
        if prepared_path is not None and prepared_path.is_file():
            selected_condition = prepared_path
            preprocessed_text = str(raw.get("condition_preprocessed", "")).strip()
            condition_preprocessed = (
                _truthy(preprocessed_text) if preprocessed_text else True
            )
        elif original_path is not None and original_path.is_file():
            selected_condition = original_path
            condition_preprocessed = False
        else:
            raise FileNotFoundError(
                f"{sample_id}: no valid condition image propagated by SS decoder"
            )

        candidates.append(
            {
                "sample_id": sample_id,
                "dataset_index": int(raw.get("dataset_index") or row_index),
                "ss_manifest": str(manifest_path),
                "coords_path": str(coords_path),
                "condition_image_path": str(original_path) if original_path else "",
                "prepared_condition_path": str(prepared_path) if prepared_path else "",
                "selected_condition_path": str(selected_condition),
                "condition_preprocessed": condition_preprocessed,
                "condition_features_path": (
                    str(condition_features_path) if condition_features_path else ""
                ),
                "rng_state_path": str(rng_state_path) if rng_state_path else "",
                "seed": raw.get("seed", ""),
                "source_stage": raw.get("source_stage", raw.get("stage", "")),
            }
        )
    if not candidates:
        raise ValueError(f"No successful SS coordinate artifacts in {manifest_path}")

    selected = _select_indices(
        len(candidates),
        num_samples=num_samples,
        seed=seed,
        indices=parse_indices(indices),
    )
    return [candidates[index] for index in selected]


def successful_slat_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Read successful SLat producer rows and resolve latent paths."""

    manifest_path = _absolute(manifest_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in enumerate(read_stage_manifest(manifest_path)):
        if _truthy(raw.get("failed", "false")):
            continue
        sample_id = (raw.get("sample_id") or "").strip()
        latent_text = (raw.get("latent_path") or "").strip()
        if not sample_id or not latent_text:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {manifest_path}")
        seen.add(sample_id)
        latent_path = _resolve_manifest_path(latent_text, manifest_path)
        if not latent_path.is_file():
            raise FileNotFoundError(f"{sample_id}: SLat artifact not found: {latent_path}")
        rows.append(
            {
                **raw,
                "sample_id": sample_id,
                "sample_index": int(raw.get("sample_index") or row_index),
                "latent_path": str(latent_path),
            }
        )
    if not rows:
        raise ValueError(f"No successful SLat artifacts in {manifest_path}")
    return rows


def load_ss_coords(path: Path, *, resolution: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "coords" not in data.files:
            raise KeyError(f"{path} is missing 'coords'; keys={data.files}")
        coords = np.asarray(data["coords"], dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3), got {coords.shape}: {path}")
    if coords.shape[0] == 0:
        raise ValueError(f"SS coordinate artifact is empty: {path}")
    if coords.min() < 0 or coords.max() >= resolution:
        raise ValueError(
            f"SS coords outside [0,{resolution - 1}] for SLat flow: "
            f"min={coords.min()}, max={coords.max()}, path={path}"
        )
    return coords


def _to_numpy(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _validate_slat_arrays(coords: np.ndarray, feats: np.ndarray, path: Path) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords shape (N,3), got {coords.shape}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: coords/feats mismatch: {coords.shape} vs {feats.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: SLat artifact is empty")
    if not np.isfinite(feats).all():
        raise ValueError(f"{path}: SLat features contain non-finite values")


def save_slat_latent(
    path: Path,
    *,
    coords: Any,
    feats: Any,
    mean: Any | None = None,
    logvar: Any | None = None,
    normalized_feats: Any | None = None,
) -> None:
    """Save a decoder-ready SLat artifact using the current dataset convention."""

    coords_array = _to_numpy(coords, dtype=np.int32)
    feats_array = _to_numpy(feats, dtype=np.float32)
    _validate_slat_arrays(coords_array, feats_array, path)
    arrays: dict[str, np.ndarray] = {
        SLAT_COORDS_KEY: coords_array,
        SLAT_FEATS_KEY: feats_array,
    }
    for key, value in (
        ("mean", mean),
        ("logvar", logvar),
        ("normalized_feats", normalized_feats),
    ):
        if value is not None:
            array = _to_numpy(value, dtype=np.float32)
            if array.shape != feats_array.shape:
                raise ValueError(
                    f"{path}: {key} shape {array.shape} != feats shape {feats_array.shape}"
                )
            arrays[key] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_slat_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        for key in (SLAT_COORDS_KEY, SLAT_FEATS_KEY):
            if key not in data.files:
                raise KeyError(f"{path} is missing {key!r}; keys={data.files}")
        coords = np.asarray(data[SLAT_COORDS_KEY], dtype=np.int32)
        feats = np.asarray(data[SLAT_FEATS_KEY], dtype=np.float32)
    _validate_slat_arrays(coords, feats, path)
    return coords, feats


def make_trellis_sparse_tensor(
    coords: Any,
    feats: Any,
    *,
    device: Any,
    dtype: Any | None = None,
):
    """Construct one current-TRELLIS SparseTensor from canonical arrays."""

    import torch
    from trellis.modules import sparse as sp

    coords_tensor = torch.as_tensor(coords, dtype=torch.int32, device=device)
    feats_tensor = torch.as_tensor(feats, device=device)
    if dtype is not None:
        feats_tensor = feats_tensor.to(dtype=dtype)
    else:
        feats_tensor = feats_tensor.float()
    if coords_tensor.ndim != 2 or coords_tensor.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3), got {tuple(coords_tensor.shape)}")
    if feats_tensor.ndim != 2 or feats_tensor.shape[0] != coords_tensor.shape[0]:
        raise ValueError(
            f"coords/feats mismatch: {tuple(coords_tensor.shape)} vs {tuple(feats_tensor.shape)}"
        )
    batch = torch.zeros((coords_tensor.shape[0], 1), dtype=torch.int32, device=device)
    return sp.SparseTensor(
        coords=torch.cat([batch, coords_tensor], dim=1),
        feats=feats_tensor,
    )


def slat_stats(coords: Any, feats: Any) -> dict[str, Any]:
    coords_array = _to_numpy(coords, dtype=np.int32)
    feats_array = _to_numpy(feats, dtype=np.float32)
    _validate_slat_arrays(coords_array, feats_array, Path("<memory>"))
    return {
        "num_points": int(coords_array.shape[0]),
        "channels": int(feats_array.shape[1]),
        "coords_min": coords_array.min(axis=0).astype(int).tolist(),
        "coords_max": coords_array.max(axis=0).astype(int).tolist(),
        "feats_mean": float(feats_array.mean()),
        "feats_std": float(feats_array.std()),
        "feats_min": float(feats_array.min()),
        "feats_max": float(feats_array.max()),
    }


def normalization_from_config(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reuse the normalization schema already consumed by current SLat eval."""

    from eval.common.slat_flow import latent_normalization_from_config

    return latent_normalization_from_config(config)


def denormalize_slat_feats(
    normalized_feats: Any,
    normalization: tuple[np.ndarray, np.ndarray] | None,
) -> Any:
    if normalization is None:
        return normalized_feats
    import torch

    mean, std = normalization
    mean_tensor = torch.as_tensor(mean, device=normalized_feats.device, dtype=normalized_feats.dtype)
    std_tensor = torch.as_tensor(std, device=normalized_feats.device, dtype=normalized_feats.dtype)
    if normalized_feats.shape[1] != mean_tensor.shape[1]:
        raise ValueError(
            f"Flow channels {normalized_feats.shape[1]} != normalization channels "
            f"{mean_tensor.shape[1]}"
        )
    return normalized_feats * std_tensor + mean_tensor


def require_nonempty_unique_samples(rows: Iterable[dict[str, Any]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not sample_ids:
        raise ValueError("No samples selected")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected samples contain duplicate sample_id values")
