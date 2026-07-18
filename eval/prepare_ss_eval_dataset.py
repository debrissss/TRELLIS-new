"""Prepare a fixed SparseStructure evaluation dataset.

The output is a normal TRELLIS dataset root with a subset metadata.csv and a
voxels symlink pointing back to the source dataset. This keeps evaluation on
the same dataset code path as training.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _ensure_clean_output_root(output_root: Path, replace: bool) -> None:
    if not output_root.exists():
        return
    if not replace:
        raise FileExistsError(f"Output root already exists: {output_root}")
    metadata_path = output_root / "metadata.csv"
    voxels_path = output_root / "voxels"
    if metadata_path.exists():
        metadata_path.unlink()
    if voxels_path.is_symlink() or voxels_path.is_file():
        voxels_path.unlink()
    elif voxels_path.exists():
        raise FileExistsError(f"Refusing to remove non-symlink voxels path: {voxels_path}")


def create_eval_dataset(
    source_root: str | Path,
    output_root: str | Path,
    num_samples: int,
    seed: int,
    min_aesthetic_score: float | None = None,
    replace: bool = False,
) -> list[str]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    metadata_path = source_root / "metadata.csv"
    voxels_dir = source_root / "voxels"

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")
    if not voxels_dir.is_dir():
        raise FileNotFoundError(f"Missing voxels directory: {voxels_dir}")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"metadata.csv must contain a sha256 column: {metadata_path}")
    if "voxelized" not in metadata.columns:
        raise ValueError(f"metadata.csv must contain a voxelized column: {metadata_path}")

    selected = metadata[_truthy_series(metadata["voxelized"])].copy()
    if min_aesthetic_score is not None:
        if "aesthetic_score" not in selected.columns:
            raise ValueError("min_aesthetic_score was requested, but metadata has no aesthetic_score column")
        selected = selected[selected["aesthetic_score"] >= min_aesthetic_score].copy()

    has_voxel = selected["sha256"].apply(lambda sha: (voxels_dir / f"{sha}.ply").is_file())
    selected = selected[has_voxel].copy()

    if len(selected) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but only {len(selected)} valid voxelized samples are available"
        )
    if len(selected) > num_samples:
        selected = selected.sample(n=num_samples, random_state=seed).sort_values("sha256").copy()

    _ensure_clean_output_root(output_root, replace=replace)
    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / "metadata.csv", index=False)

    link_target = os.path.relpath(voxels_dir.resolve(), start=output_root.resolve())
    os.symlink(link_target, output_root / "voxels")
    return selected["sha256"].tolist()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, help="Source TRELLIS dataset root containing metadata.csv and voxels/")
    parser.add_argument("--output_root", required=True, help="Output mini dataset root")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of fixed evaluation samples")
    parser.add_argument("--seed", type=int, default=20260718, help="Random seed for sample selection")
    parser.add_argument("--min_aesthetic_score", type=float, default=None, help="Optional minimum aesthetic_score filter")
    parser.add_argument("--replace", action="store_true", help="Replace an existing output metadata/symlink")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    selected = create_eval_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        num_samples=args.num_samples,
        seed=args.seed,
        min_aesthetic_score=args.min_aesthetic_score,
        replace=args.replace,
    )
    print(f"Wrote {len(selected)} samples to {Path(args.output_root) / 'metadata.csv'}")
    print(f"First sample: {selected[0]}")
    print(f"Last sample: {selected[-1]}")


if __name__ == "__main__":
    main()
