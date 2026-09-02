"""Build the FaceScan paired dataset used by SS Flow ControlNet.

The generated layout follows the split-root convention used by
``datasets/Facescape`` while keeping ControlNet inputs and supervision assets
in different directories.  Large PLY/PNG files are linked instead of copied.

This script does not run the SS Encoder.  Target latents must be generated
from ``target_voxels`` afterwards and stored under
``ss_latents/<latent_model>/<id>.npz`` before training can consume a sample.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_LATENT_MODEL = (
    "ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8_step0002000"
)
TEST_MARKER = "model.ply缺失"


def _subject_id(directory: Path) -> Optional[str]:
    """Return the leading numeric folder id and ignore utility directories."""
    match = re.match(r"^(\d+)", directory.name)
    return match.group(1) if match else None


def _relative_symlink(source: Path, destination: Path) -> None:
    """Create/update one relative symlink without copying a large asset."""
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) == target:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink dataset asset: {destination}"
        )
    destination.symlink_to(target)


def _ply_vertex_count(path: Path) -> int:
    """Read only the PLY header and return its declared vertex count."""
    if not path.is_file():
        return 0
    with path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY end_header not found: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                return int(text.rsplit(" ", 1)[-1])


def _bool(value: bool) -> str:
    # pandas recognizes these strings as boolean values when metadata is read.
    return "True" if value else "False"


def _build_record(
    source_dir: Path,
    split_root: Path,
    split: str,
    instance: str,
    latent_model: str,
) -> Dict[str, object]:
    image = source_dir / "model" / "up_normal.png"
    control_mesh = (
        source_dir
        / "align_to_standard_filter"
        / "merged_normalized_mesh.ply"
    )
    target_mesh = source_dir / "model_normalized_nocolor.ply"
    control_voxel = (
        source_dir
        / "align_to_standard_filter"
        / "merged_normalized_voxelized.ply"
    )
    target_voxel = source_dir / "model_normalized_voxelized.ply"

    # ControlNet 改动：条件 mesh/voxel 与监督 mesh/voxel 永久分目录，避免
    # encode_ss_latent 一类脚本误把 partial control 编码成 flow 的 x_0。
    _relative_symlink(image, split_root / "renders_cond" / instance / "up_normal.png")
    _relative_symlink(control_mesh, split_root / "control_meshes" / f"{instance}.ply")
    _relative_symlink(target_mesh, split_root / "target_meshes" / f"{instance}.ply")
    _relative_symlink(control_voxel, split_root / "control_voxels" / f"{instance}.ply")
    _relative_symlink(target_voxel, split_root / "target_voxels" / f"{instance}.ply")

    latent = split_root / "ss_latents" / latent_model / f"{instance}.npz"
    has_image = image.is_file()
    has_control_mesh = control_mesh.is_file()
    has_target_mesh = target_mesh.is_file()
    has_control_voxel = control_voxel.is_file()
    has_target_voxel = target_voxel.is_file()

    return {
        "sha256": instance,
        "local_path": str(source_dir.resolve()),
        "aesthetic_score": 5.0,
        "rendered": _bool(False),
        # Keep the standard FaceScape field: here it describes target voxels.
        "voxelized": _bool(has_target_voxel),
        "num_voxels": _ply_vertex_count(target_voxel),
        "cond_rendered": _bool(has_image),
        "captions": f"A normalized face scan with subject id {instance}",
        "split": split,
        "control_mesh": _bool(has_control_mesh),
        "control_voxelized": _bool(has_control_voxel),
        "num_control_voxels": _ply_vertex_count(control_voxel),
        "target_mesh": _bool(has_target_mesh),
        "target_voxelized": _bool(has_target_voxel),
        f"ss_latent_{latent_model}": _bool(latent.is_file()),
    }


def _write_metadata(path: Path, records: Iterable[Dict[str, object]]) -> None:
    records = list(records)
    if not records:
        raise ValueError(f"No records available for {path.parent.name} split")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_dataset(source_root: Path, output_root: Path, latent_model: str) -> None:
    split_records = {"train": [], "test": []}
    seen_ids = set()

    for source_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        instance = _subject_id(source_dir)
        if instance is None:
            continue
        if instance in seen_ids:
            raise ValueError(f"Duplicate FaceScan folder id: {instance}")
        seen_ids.add(instance)

        # 用户指定：名称标注 model.ply 缺失的两个样本作为测试集；其余
        # 数字 id 样本作为训练集。测试 metadata 保留缺失状态，不伪造 GT。
        split = "test" if TEST_MARKER in source_dir.name else "train"
        split_root = output_root / split
        record = _build_record(
            source_dir,
            split_root,
            split,
            instance,
            latent_model,
        )
        split_records[split].append(record)

    for split, records in split_records.items():
        split_root = output_root / split
        # Create the complete standard directory contract even before GPU latent
        # preprocessing has populated it.
        for directory in (
            "renders_cond",
            "control_meshes",
            "target_meshes",
            "control_voxels",
            "target_voxels",
        ):
            (split_root / directory).mkdir(parents=True, exist_ok=True)
        (split_root / "ss_latents" / latent_model).mkdir(
            parents=True,
            exist_ok=True,
        )
        _write_metadata(split_root / "metadata.csv", records)
        print(f"{split}: {len(records)} records -> {split_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FaceScan train/test roots for SS Flow ControlNet"
    )
    parser.add_argument(
        "--source_root",
        default="face_scan_test_data",
        help="Root containing one folder per FaceScan id.",
    )
    parser.add_argument(
        "--output_root",
        default="datasets/FaceScan_ControlNet",
        help="Output root containing train/ and test/.",
    )
    parser.add_argument(
        "--latent_model",
        default=DEFAULT_LATENT_MODEL,
        help="Target SS latent directory/metadata suffix.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FaceScan source root not found: {source_root}")
    build_dataset(source_root, output_root, args.latent_model)


if __name__ == "__main__":
    main()
