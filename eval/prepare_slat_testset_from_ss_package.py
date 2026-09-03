#!/usr/bin/env python3
"""Build a one-sample SLat test set from an exported SS-stage package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import write_csv, write_json
from eval.common.ss_inference import save_image_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--condition-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_visualization_coords(path: Path) -> np.ndarray:
    vertices = PlyData.read(path)["vertex"].data
    points = np.stack([vertices[axis] for axis in ("x", "y", "z")], axis=1)
    return np.floor((points + 0.5) * 64 + 1e-6).astype(np.int32)


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    condition_image = args.condition_image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_manifest_path = package_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    primary_path = package_dir / source_manifest["primary_file"]
    sparse_ply_path = package_dir / "ss_generated_sparse_structure.ply"
    package_image_path = package_dir / "input_up_normal.png"
    for path in [primary_path, sparse_ply_path, package_image_path, condition_image]:
        if not path.is_file():
            raise FileNotFoundError(path)

    primary_sha = sha256(primary_path)
    if primary_sha != source_manifest["sha256"]:
        raise ValueError(
            f"Package checksum mismatch: manifest={source_manifest['sha256']}, actual={primary_sha}"
        )
    condition_sha = sha256(condition_image)
    package_image_sha = sha256(package_image_path)
    if condition_sha != package_image_sha:
        raise ValueError(
            "The requested condition image does not match the image used to produce the SS package: "
            f"requested={condition_sha}, package={package_image_sha}"
        )

    tensors = load_file(primary_path, device="cpu")
    missing = {"coords", "cond", "neg_cond"}.difference(tensors)
    if missing:
        raise KeyError(f"SS package is missing tensors: {sorted(missing)}")
    coords4 = tensors["coords"].detach().cpu().numpy().astype(np.int32, copy=False)
    if coords4.ndim != 2 or coords4.shape[1] != 4:
        raise ValueError(f"Expected package coords shape (N,4), got {coords4.shape}")
    if not np.all(coords4[:, 0] == 0):
        raise ValueError(f"Expected a single batch with index 0, got {np.unique(coords4[:, 0])}")
    coords = coords4[:, 1:]
    if coords.shape[0] == 0 or coords.min() < 0 or coords.max() >= 64:
        raise ValueError(f"Coords outside the SLat resolution-64 domain: {coords.shape}")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("SS package contains duplicate sparse coordinates")
    visual_coords = load_visualization_coords(sparse_ply_path)
    if {tuple(row) for row in visual_coords} != {tuple(row) for row in coords}:
        raise ValueError("ss_generated_sparse_structure.ply does not match package coords")

    sample_id = primary_sha
    coords_path = output_dir / "ss_coords" / f"{sample_id}.npz"
    features_path = output_dir / "condition_features" / f"{sample_id}.npz"
    condition_path = output_dir / "renders_cond" / sample_id / "up_normal.png"
    sparse_output_path = output_dir / "sparse_structures" / f"{sample_id}.ply"
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coords_path, coords=coords)
    save_image_condition(
        features_path,
        {"cond": tensors["cond"], "neg_cond": tensors["neg_cond"]},
    )
    copy(condition_image, condition_path)
    copy(sparse_ply_path, sparse_output_path)
    copy(primary_path, output_dir / "source_package" / primary_path.name)
    copy(source_manifest_path, output_dir / "source_package" / "manifest.json")

    seed = int(source_manifest.get("recommended_slat_seed", 42))
    write_csv(
        output_dir / "ss_manifest.csv",
        [
            {
                "stage": "ss_controlnet_package",
                "source_stage": "ss_flow",
                "sample_id": sample_id,
                "dataset_index": 0,
                "coords_path": coords_path.relative_to(output_dir),
                "condition_image_path": condition_path.relative_to(output_dir),
                "prepared_condition_path": condition_path.relative_to(output_dir),
                "condition_preprocessed": True,
                "condition_features_path": features_path.relative_to(output_dir),
                "rng_state_path": "",
                "seed": seed,
                "failed": False,
                "error": "",
            }
        ],
    )
    write_csv(
        output_dir / "metadata.csv",
        [
            {
                "sha256": sample_id,
                "local_path": sparse_output_path.relative_to(output_dir),
                "rendered": True,
                "voxelized": True,
                "num_voxels": int(coords.shape[0]),
                "cond_rendered": True,
                "split": "test",
                "source_format": source_manifest.get("format", ""),
                "recommended_slat_seed": seed,
            }
        ],
    )
    dataset_manifest = {
        "format": "trellis_slat_coarse_structure_testset_v1",
        "sample_id": sample_id,
        "num_samples": 1,
        "resolution": 64,
        "num_coords": int(coords.shape[0]),
        "coord_min": coords.min(axis=0).astype(int).tolist(),
        "coord_max": coords.max(axis=0).astype(int).tolist(),
        "condition_image_sha256": condition_sha,
        "source_package_sha256": primary_sha,
        "source_package_format": source_manifest.get("format"),
        "recommended_slat_seed": seed,
        "image_preprocess": source_manifest.get("image_preprocess"),
        "artifacts": {
            "ss_manifest": "ss_manifest.csv",
            "coords": str(coords_path.relative_to(output_dir)),
            "condition_features": str(features_path.relative_to(output_dir)),
            "condition_image": str(condition_path.relative_to(output_dir)),
            "sparse_structure_ply": str(sparse_output_path.relative_to(output_dir)),
            "source_package": str(
                (output_dir / "source_package" / primary_path.name).relative_to(output_dir)
            ),
        },
        "continuation": {
            "entrypoint": "eval.slat_inference_pipeline",
            "ss_manifest": "ss_manifest.csv",
            "rng": "seed_fallback",
            "note": "The package contains no post-SS RNG state; use recommended_slat_seed=42 as specified by its continuation contract.",
        },
    }
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    print(json.dumps(dataset_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
