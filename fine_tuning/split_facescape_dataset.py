"""
Split a merged FaceScape TRELLIS dataset into train/val/test subsets.

The source dataset is left untouched. By default this script creates:

  <dataset_dir>/train/metadata.csv
  <dataset_dir>/train/latents/<latent_model>/<sha256>.npz
  <dataset_dir>/train/renders_cond/<sha256>/...

and the same layout for val and test.
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_subject_id(local_path: str) -> int | None:
    """Extract the FaceScape subject id from paths like .../359/9_mouth_right.ply."""
    if pd.isna(local_path):
        return None

    parts = Path(str(local_path)).parts
    if not parts:
        return None

    for part in reversed(parts[:-1]):
        if re.fullmatch(r"\d+", part):
            return int(part)
    return None


def same_file(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def copy_file(src: Path, dst: Path, *, overwrite: str, dry_run: bool) -> str:
    if dst.exists():
        if same_file(src, dst):
            return "skipped_same"
        if overwrite == "error":
            return "conflict"
        if overwrite == "skip":
            return "skipped_conflict"
        if overwrite == "replace" and not dry_run:
            dst.unlink()

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "copied"


def copy_tree(src: Path, dst: Path, *, overwrite: str, dry_run: bool) -> dict[str, int]:
    stats = {
        "copied": 0,
        "skipped_same": 0,
        "skipped_conflict": 0,
        "conflict": 0,
    }
    if not src.exists():
        stats["missing"] = 1
        return stats

    files = [path for path in src.rglob("*") if path.is_file()]
    for file_path in files:
        rel = file_path.relative_to(src)
        status = copy_file(file_path, dst / rel, overwrite=overwrite, dry_run=dry_run)
        stats[status] += 1
    return stats


def add_stats(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def discover_latent_models(dataset_dir: Path, metadata: pd.DataFrame) -> list[str]:
    models = set()

    for column in metadata.columns:
        if column.startswith("latent_"):
            models.add(column.removeprefix("latent_"))

    latents_dir = dataset_dir / "latents"
    if latents_dir.exists():
        models.update(path.name for path in latents_dir.iterdir() if path.is_dir())

    return sorted(models)


def resolve_latent_column(metadata: pd.DataFrame, requested_column: str | None) -> str:
    if requested_column is not None:
        if requested_column not in metadata.columns:
            raise ValueError(f"metadata.csv does not contain requested latent column: {requested_column}")
        return requested_column

    latent_columns = [column for column in metadata.columns if column.startswith("latent_")]
    if len(latent_columns) == 1:
        return latent_columns[0]
    if not latent_columns:
        raise ValueError("metadata.csv does not contain any latent_* column")
    raise ValueError(
        "metadata.csv contains multiple latent_* columns. "
        f"Please pass --latent_column explicitly. Candidates: {latent_columns}"
    )


def filter_ready_metadata(
    metadata: pd.DataFrame,
    *,
    latent_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "cond_rendered" not in metadata.columns:
        raise ValueError("metadata.csv must contain cond_rendered column")

    ready_mask = metadata[latent_column].apply(truthy) & metadata["cond_rendered"].apply(truthy)
    ready = metadata[ready_mask].copy()
    excluded = metadata[~ready_mask].copy()
    return ready, excluded


def split_metadata(
    metadata: pd.DataFrame,
    *,
    anchor_subject: int,
    test_count: int,
    val_count: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    metadata = metadata.copy()
    local_path = metadata["local_path"]
    missing_local_path = local_path.isna() | local_path.astype(str).str.strip().eq("")
    no_split_local_path = metadata[missing_local_path].copy()
    split_candidates = metadata[~missing_local_path].copy()

    split_candidates["_facescape_subject_id"] = split_candidates["local_path"].apply(parse_subject_id)

    unparsed = split_candidates[split_candidates["_facescape_subject_id"].isna()]
    if not unparsed.empty:
        examples = unparsed["local_path"].head(5).tolist()
        raise ValueError(
            "Failed to parse FaceScape subject id from local_path for "
            f"{len(unparsed)} rows. Examples: {examples}"
        )

    test_min = anchor_subject - test_count + 1
    test_max = anchor_subject
    val_max = test_min - 1
    val_min = val_max - val_count + 1

    test_mask = split_candidates["_facescape_subject_id"].between(test_min, test_max)
    val_mask = split_candidates["_facescape_subject_id"].between(val_min, val_max)
    train_mask = ~(test_mask | val_mask)

    drop_internal = ["_facescape_subject_id"]
    train = pd.concat(
        [
            split_candidates[train_mask].drop(columns=drop_internal),
            no_split_local_path,
        ],
        ignore_index=True,
    )
    split = {
        "train": train,
        "val": split_candidates[val_mask].drop(columns=drop_internal),
        "test": split_candidates[test_mask].drop(columns=drop_internal),
    }
    return split, no_split_local_path


def copy_subset(
    *,
    dataset_dir: Path,
    subset_dir: Path,
    subset_name: str,
    metadata: pd.DataFrame,
    latent_models: list[str],
    overwrite: str,
    dry_run: bool,
    allow_missing: bool,
) -> dict[str, dict[str, int]]:
    totals = {
        "latents": {},
        "renders_cond": {},
    }
    missing = []

    iterator = metadata["sha256"].astype(str).tolist()
    for sha256 in tqdm(iterator, desc=f"Copying {subset_name}", leave=False):
        render_src = dataset_dir / "renders_cond" / sha256
        render_dst = subset_dir / "renders_cond" / sha256
        if not render_src.exists():
            missing.append(str(render_src))
        else:
            add_stats(
                totals["renders_cond"],
                copy_tree(render_src, render_dst, overwrite=overwrite, dry_run=dry_run),
            )

        for latent_model in latent_models:
            latent_src = dataset_dir / "latents" / latent_model / f"{sha256}.npz"
            latent_dst = subset_dir / "latents" / latent_model / f"{sha256}.npz"
            if not latent_src.exists():
                missing.append(str(latent_src))
                continue
            status = copy_file(latent_src, latent_dst, overwrite=overwrite, dry_run=dry_run)
            totals["latents"][status] = totals["latents"].get(status, 0) + 1

    if missing and not allow_missing:
        examples = missing[:10]
        raise FileNotFoundError(
            f"{subset_name} has {len(missing)} missing source files/directories. "
            f"Examples: {examples}. Use --allow_missing to write partial splits."
        )

    if not dry_run:
        subset_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(subset_dir / "metadata.csv", index=False)

    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a merged FaceScape TRELLIS dataset by subject id from metadata local_path."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/root/autodl-tmp/TRELLIS/datasets/Facescape"),
        help="Merged FaceScape dataset root containing metadata.csv, latents, and renders_cond.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Directory where train/val/test will be created. Defaults to dataset_dir.",
    )
    parser.add_argument(
        "--anchor_subject",
        type=int,
        default=359,
        help="Highest subject id in the test split. Defaults to 359.",
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=36,
        help="Number of subject ids assigned to test, counting backwards from anchor_subject.",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=36,
        help="Number of subject ids assigned to val immediately before the test range.",
    )
    parser.add_argument(
        "--latent_column",
        type=str,
        default=None,
        help="Metadata latent_* column required to be true. Auto-detected when exactly one latent_* column exists.",
    )
    parser.add_argument(
        "--overwrite",
        choices=["error", "skip", "replace"],
        default="error",
        help="How to handle existing destination files with different sizes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned split without copying files or writing metadata.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Allow missing latent/render files and still write partial split metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_root = args.output_root or dataset_dir
    metadata_path = dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    for column in ["sha256", "local_path"]:
        if column not in metadata.columns:
            raise ValueError(f"metadata.csv must contain {column} column: {metadata_path}")

    metadata["sha256"] = metadata["sha256"].astype(str)
    latent_column = resolve_latent_column(metadata, args.latent_column)
    latent_model = latent_column.removeprefix("latent_")
    ready_metadata, excluded_metadata = filter_ready_metadata(metadata, latent_column=latent_column)
    print(f"[INFO] source metadata rows: {len(metadata)}")
    print(f"[INFO] ready rows for splitting: {len(ready_metadata)} ({latent_column}=true and cond_rendered=true)")
    print(f"[INFO] excluded rows not ready: {len(excluded_metadata)}")

    if not excluded_metadata.empty and not args.dry_run:
        excluded_path = output_root / "split_excluded_not_ready.csv"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_metadata.to_csv(excluded_path, index=False)
        print(f"[INFO] excluded rows report written to {excluded_path}")

    split, no_split_local_path = split_metadata(
        ready_metadata,
        anchor_subject=args.anchor_subject,
        test_count=args.test_count,
        val_count=args.val_count,
    )
    latent_models = [latent_model]
    if not (dataset_dir / "latents" / latent_model).exists():
        raise FileNotFoundError(f"Latent model directory not found: {dataset_dir / 'latents' / latent_model}")

    test_min = args.anchor_subject - args.test_count + 1
    val_max = test_min - 1
    val_min = val_max - args.val_count + 1
    print(f"[INFO] val subject range: {val_min}-{val_max}")
    print(f"[INFO] test subject range: {test_min}-{args.anchor_subject}")
    print(f"[INFO] latent model: {latent_model}")

    for subset_name in ["train", "val", "test"]:
        subset_metadata = split[subset_name]
        subject_count = subset_metadata["local_path"].apply(parse_subject_id).nunique()
        print(f"[INFO] {subset_name}: {len(subset_metadata)} rows, {subject_count} subjects")

    if not no_split_local_path.empty:
        print(f"[WARN] rows with empty local_path assigned to train: {len(no_split_local_path)}")
        if not args.dry_run:
            empty_local_path_report = output_root / "split_empty_local_path_assigned_to_train.csv"
            empty_local_path_report.parent.mkdir(parents=True, exist_ok=True)
            no_split_local_path.to_csv(empty_local_path_report, index=False)
            print(f"[WARN] empty local_path train report written to {empty_local_path_report}")

    for subset_name, subset_metadata in split.items():
        subset_dir = output_root / subset_name
        stats = copy_subset(
            dataset_dir=dataset_dir,
            subset_dir=subset_dir,
            subset_name=subset_name,
            metadata=subset_metadata,
            latent_models=latent_models,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            allow_missing=args.allow_missing,
        )
        print(f"[SUMMARY] {subset_name}: {stats}")

    if args.dry_run:
        print("[SUMMARY] dry run: no files were copied and no metadata.csv files were written")
    else:
        print(f"[SUMMARY] split datasets written under {output_root}")


if __name__ == "__main__":
    main()
