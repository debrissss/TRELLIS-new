"""Shared artifact, dataset, and model-loading helpers for split SS inference.

This module only contains orchestration and IO glue. Sparse-structure model
forward passes, image conditioning, flow sampling, and decoding remain owned by
the existing TRELLIS/Stable3DGen implementations.
"""

from __future__ import annotations

import csv
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval.common.dataset import TRUE_VALUES, read_metadata_rows
from eval.common.io import safe_tag, write_csv, write_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
INPUT_MANIFEST_FIELDS = [
    "sample_id",
    "dataset_index",
    "voxel_path",
    "condition_image_path",
]
LATENT_KEY = "z_s"
TORCH_CPU_RNG_STATE_KEY = "torch_cpu_rng_state"
IMAGE_CONDITION_KEYS = ("cond", "neg_cond")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def parse_indices(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def list_condition_images(data_dir: Path, sample_id: str) -> list[Path]:
    condition_dir = data_dir / "renders_cond" / sample_id
    if not condition_dir.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in condition_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _select_candidate_indices(
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


def select_dataset_inputs(
    data_dir: Path,
    *,
    num_samples: int,
    seed: int,
    indices: list[int] | None,
    require_voxel: bool,
    require_condition: bool,
) -> list[dict[str, Any]]:
    """Select deterministic dataset inputs and freeze condition-image choices."""

    data_dir = _absolute(data_dir)
    metadata_rows = read_metadata_rows(data_dir / "metadata.csv")
    candidates: list[dict[str, Any]] = []
    for metadata_index, row in enumerate(metadata_rows):
        sample_id = row["sha256"].strip()
        if not sample_id:
            continue
        if require_voxel and "voxelized" in row and not _truthy(row["voxelized"]):
            continue
        if require_condition and "cond_rendered" in row and not _truthy(row["cond_rendered"]):
            continue

        voxel_path = (data_dir / "voxels" / f"{sample_id}.ply").resolve()
        condition_images = list_condition_images(data_dir, sample_id)
        if require_voxel and not voxel_path.is_file():
            continue
        if require_condition and not condition_images:
            continue
        candidates.append(
            {
                "sample_id": sample_id,
                "dataset_index": metadata_index,
                "voxel_path": str(voxel_path) if voxel_path.is_file() else "",
                "_condition_images": condition_images,
            }
        )

    if not candidates:
        requirements = []
        if require_voxel:
            requirements.append("voxel")
        if require_condition:
            requirements.append("condition image")
        raise ValueError(f"No dataset samples satisfy requirements: {', '.join(requirements)}")

    selected_indices = _select_candidate_indices(
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
        condition_image = condition_rng.choice(condition_images) if condition_images else None
        candidate["condition_image_path"] = str(condition_image) if condition_image else ""
        selected.append(candidate)
    return selected


def read_input_manifest(
    manifest_path: Path,
    *,
    require_voxel: bool,
    require_condition: bool,
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

        voxel_text = (raw.get("voxel_path") or "").strip()
        condition_text = (
            raw.get("condition_image_path")
            or raw.get("normal_image")
            or raw.get("selected_normal")
            or ""
        ).strip()
        voxel_path = _resolve_manifest_path(voxel_text, manifest_path) if voxel_text else None
        condition_path = _resolve_manifest_path(condition_text, manifest_path) if condition_text else None
        if require_voxel and (voxel_path is None or not voxel_path.is_file()):
            raise FileNotFoundError(f"{sample_id}: voxel_path is missing or invalid: {voxel_path}")
        if require_condition and (condition_path is None or not condition_path.is_file()):
            raise FileNotFoundError(
                f"{sample_id}: condition_image_path is missing or invalid: {condition_path}"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": int(raw.get("dataset_index") or row_index),
                "voxel_path": str(voxel_path) if voxel_path else "",
                "condition_image_path": str(condition_path) if condition_path else "",
            }
        )
    return rows


def resolve_input_records(
    *,
    data_dir: Path | None,
    input_manifest: Path | None,
    num_samples: int,
    seed: int,
    indices: str | None,
    require_voxel: bool,
    require_condition: bool,
) -> list[dict[str, Any]]:
    if (data_dir is None) == (input_manifest is None):
        raise ValueError("Exactly one of --data_dir or --input_manifest must be provided")
    if input_manifest is not None:
        return read_input_manifest(
            input_manifest,
            require_voxel=require_voxel,
            require_condition=require_condition,
        )
    assert data_dir is not None
    return select_dataset_inputs(
        data_dir,
        num_samples=num_samples,
        seed=seed,
        indices=parse_indices(indices),
        require_voxel=require_voxel,
        require_condition=require_condition,
    )


def write_input_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, fieldnames=INPUT_MANIFEST_FIELDS)


def read_stage_manifest(manifest_path: Path) -> list[dict[str, str]]:
    manifest_path = _absolute(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Stage manifest is empty: {manifest_path}")
    return rows


def successful_latent_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Read a producer manifest and resolve successful latent paths."""

    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(read_stage_manifest(manifest_path)):
        if _truthy(raw.get("failed", "false")):
            continue
        sample_id = (raw.get("sample_id") or "").strip()
        latent_text = (raw.get("latent_path") or "").strip()
        if not sample_id or not latent_text:
            continue
        latent_path = _resolve_manifest_path(latent_text, _absolute(manifest_path))
        if not latent_path.is_file():
            raise FileNotFoundError(f"{sample_id}: latent artifact not found: {latent_path}")
        rows.append(
            {
                **raw,
                "sample_id": sample_id,
                "sample_index": int(raw.get("sample_index") or row_index),
                "latent_path": str(latent_path),
            }
        )
    if not rows:
        raise ValueError(f"No successful latent artifacts in {manifest_path}")
    return rows


def require_device(device_name: str):
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    return torch.device(device_name)


def _checkpoint_prefix_exists(checkpoint: str) -> bool:
    path = Path(checkpoint).expanduser()
    return Path(f"{path}.json").is_file() and Path(f"{path}.safetensors").is_file()


def _unwrap_state_dict(state: Any, checkpoint: str) -> dict[str, Any]:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {checkpoint}")
    return state


def load_configured_model(
    config: dict[str, Any],
    *,
    model_key: str,
    checkpoint: str,
    device: Any,
    models_module: Any | None = None,
    model_aliases: dict[str, str] | None = None,
):
    """Load one model only, from a training .pt or pretrained model prefix."""

    import torch

    if models_module is None:
        from trellis import models as models_module

    aliases = model_aliases or {}

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        model_spec = config.get("models", {}).get(model_key)
        if model_spec is None:
            raise KeyError(f"Config is missing models.{model_key}")
        model_name = aliases.get(model_spec["name"], model_spec["name"])
        model = getattr(models_module, model_name)(**model_spec["args"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(_unwrap_state_dict(state, checkpoint), strict=True)
    elif checkpoint_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    elif _checkpoint_prefix_exists(checkpoint) or "/" in checkpoint:
        model = models_module.from_pretrained(checkpoint)
    else:
        raise FileNotFoundError(
            f"Checkpoint is neither a .pt file nor a pretrained prefix: {checkpoint}"
        )

    if getattr(device, "type", None) == "cpu" and hasattr(model, "convert_to_fp32"):
        model.convert_to_fp32()
    model = model.to(device)
    model.eval()
    return model


def load_voxel_grid(voxel_path: Path, resolution: int):
    """Use the current dataset-toolkit voxel-to-grid convention."""

    import torch
    import utils3d

    position = utils3d.io.read_ply(str(voxel_path))[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    grid[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return grid


def _tensor_array(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def save_ss_latent(
    path: Path,
    z_s: Any,
    *,
    mean: Any | None = None,
    logvar: Any | None = None,
) -> None:
    """Save the canonical split-inference SS latent artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {LATENT_KEY: _tensor_array(z_s)}
    if mean is not None:
        arrays["mean"] = _tensor_array(mean)
    if logvar is not None:
        arrays["logvar"] = _tensor_array(logvar)
    np.savez_compressed(path, **arrays)


def load_ss_latent(path: Path) -> tuple[np.ndarray, str]:
    """Load canonical z_s, while accepting current dataset-toolkit mean files."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if LATENT_KEY in data.files:
            key = LATENT_KEY
        elif "mean" in data.files:
            key = "mean"
        else:
            raise KeyError(f"{path} has neither {LATENT_KEY!r} nor 'mean'; keys={data.files}")
        latent = np.asarray(data[key], dtype=np.float32)
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(f"Expected SS latent shape (C,D,H,W), got {latent.shape}: {path}")
    if not np.isfinite(latent).all():
        raise ValueError(f"SS latent contains non-finite values: {path}")
    return latent, key


def save_torch_cpu_rng_state(path: Path, state: Any) -> None:
    """Persist the default CPU generator state across split inference stages."""

    state_array = state.detach().cpu().numpy().astype(np.uint8, copy=False)
    if state_array.ndim != 1 or state_array.size == 0:
        raise ValueError(f"Invalid torch CPU RNG state shape: {state_array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{TORCH_CPU_RNG_STATE_KEY: state_array})


def load_torch_cpu_rng_state(path: Path):
    """Load a CPU generator state saved by :func:`save_torch_cpu_rng_state`."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if TORCH_CPU_RNG_STATE_KEY not in data.files:
            raise KeyError(
                f"{path} is missing {TORCH_CPU_RNG_STATE_KEY!r}; keys={data.files}"
            )
        state_array = np.asarray(data[TORCH_CPU_RNG_STATE_KEY], dtype=np.uint8)
    if state_array.ndim != 1 or state_array.size == 0:
        raise ValueError(f"Invalid torch CPU RNG state shape in {path}: {state_array.shape}")
    return torch.from_numpy(state_array.copy())


def save_image_condition(path: Path, condition: dict[str, Any]) -> None:
    """Save the DINO condition shared by Stable3DGen's SS and SLat stages."""

    arrays: dict[str, np.ndarray] = {}
    for key in IMAGE_CONDITION_KEYS:
        if key not in condition:
            raise KeyError(f"Image condition is missing {key!r}")
        array = _tensor_array(condition[key])
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError(f"Invalid image condition {key} shape/values: {array.shape}")
        arrays[key] = array
    if arrays["cond"].shape != arrays["neg_cond"].shape:
        raise ValueError(
            "cond/neg_cond shapes differ: "
            f"{arrays['cond'].shape} vs {arrays['neg_cond'].shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_image_condition(path: Path, *, device: Any) -> dict[str, Any]:
    """Load a DINO condition artifact onto one inference device."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in IMAGE_CONDITION_KEYS:
            if key not in data.files:
                raise KeyError(f"{path} is missing {key!r}; keys={data.files}")
            array = np.asarray(data[key], dtype=np.float32)
            if array.ndim != 3 or not np.isfinite(array).all():
                raise ValueError(
                    f"Invalid image condition {key} in {path}: {array.shape}"
                )
            arrays[key] = array
    if arrays["cond"].shape != arrays["neg_cond"].shape:
        raise ValueError(
            f"{path}: cond/neg_cond shapes differ: "
            f"{arrays['cond'].shape} vs {arrays['neg_cond'].shape}"
        )
    return {
        key: torch.from_numpy(array.copy()).to(device)
        for key, array in arrays.items()
    }


def latent_stats(tensor: Any) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean().cpu()),
        "std": float(value.std().cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
    }


def sample_output_dir(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "samples" / safe_tag(sample_id)


def namespace_to_jsonable(args: Namespace) -> dict[str, Any]:
    return value_to_jsonable(vars(args))


def value_to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): value_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [value_to_jsonable(item) for item in value]
    return value


def write_stage_result(
    output_dir: Path,
    *,
    stage: str,
    args: Namespace,
    rows: list[dict[str, Any]],
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = [row for row in rows if bool(row.get("failed"))]
    summary: dict[str, Any] = {
        "stage": stage,
        "output_dir": str(output_dir.resolve()),
        "num_records": len(rows),
        "successful_count": len(rows) - len(failures),
        "failed_count": len(failures),
        "args": namespace_to_jsonable(args),
    }
    if extra_summary:
        summary.update(value_to_jsonable(extra_summary))
    write_csv(output_dir / "manifest.csv", rows)
    write_json(output_dir / "failed_samples.json", failures)
    write_json(output_dir / "summary.json", summary)
    return summary


def require_nonempty_unique_samples(rows: Iterable[dict[str, Any]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not sample_ids:
        raise ValueError("No samples selected")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected samples contain duplicate sample_id values")
