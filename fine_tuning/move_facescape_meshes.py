"""Move normalized FaceScape mesh.ply files into train/test render folders.

The source layout is expected to be:

  /root/autodl-tmp/normalized_mesh/normalized_closed_shapes_meshlib/renders/<sha256>/mesh.ply

Meshes are moved to:

  <dataset_dir>/<train|test>/renders/<sha256>/mesh.ply

Only mesh.ply is moved. transforms.json and any other source files are left in
place.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_SOURCE_RENDERS = Path(
    "/root/autodl-tmp/normalized_mesh/normalized_closed_shapes_meshlib/renders"
)
DEFAULT_DATASET_DIR = Path("/root/autodl-tmp/TRELLIS-new/datasets/Facescape")
SPLITS = ("train", "test")


def read_sha256s(metadata_path: Path) -> list[str]:
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sha256" not in (reader.fieldnames or []):
            raise ValueError(f"{metadata_path} does not contain a sha256 column")
        return [row["sha256"].strip() for row in reader if row.get("sha256", "").strip()]


def same_file_size(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def build_split_index(dataset_dir: Path) -> dict[str, str]:
    sha_to_split: dict[str, str] = {}
    for split in SPLITS:
        metadata_path = dataset_dir / split / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        for sha256 in read_sha256s(metadata_path):
            previous = sha_to_split.get(sha256)
            if previous is not None and previous != split:
                raise ValueError(f"{sha256} appears in both {previous} and {split}")
            sha_to_split[sha256] = split
    return sha_to_split


def move_meshes(
    *,
    source_renders: Path,
    dataset_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, int]:
    sha_to_split = build_split_index(dataset_dir)
    stats = {
        "planned": 0,
        "moved": 0,
        "already_present": 0,
        "missing_source": 0,
        "skipped_existing": 0,
        "replaced_existing": 0,
    }

    for sha256, split in sorted(sha_to_split.items()):
        src = source_renders / sha256 / "mesh.ply"
        dst = dataset_dir / split / "renders" / sha256 / "mesh.ply"
        if not src.exists():
            if dst.exists():
                stats["already_present"] += 1
                continue
            stats["missing_source"] += 1
            continue

        stats["planned"] += 1
        if dst.exists():
            if same_file_size(src, dst) or not overwrite:
                stats["skipped_existing"] += 1
                continue
            stats["replaced_existing"] += 1
            if not dry_run:
                dst.unlink()

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        stats["moved"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move FaceScape mesh.ply files into train/test renders folders."
    )
    parser.add_argument(
        "--source-renders",
        type=Path,
        default=DEFAULT_SOURCE_RENDERS,
        help=f"Source renders directory. Default: {DEFAULT_SOURCE_RENDERS}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Facescape dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be moved without changing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination mesh.ply when its size differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = move_meshes(
        source_renders=args.source_renders,
        dataset_dir=args.dataset_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    mode = "DRY RUN" if args.dry_run else "MOVE"
    print(f"{mode} complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
