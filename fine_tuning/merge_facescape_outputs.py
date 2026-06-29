"""
Merge sharded FaceScape preprocessing outputs into one TRELLIS training root.

Expected source layout under --source_root by default:
  preprocess_facescape_1/latents_1/
  preprocess_facescape_1/renders_cond_1/
  preprocess_facescape_1/metadata_1.csv
  preprocess_facescape_2/latents_2/
  ...

Output layout:
  <output_dir>/latents/<latent_model>/<sha256>.npz
  <output_dir>/renders_cond/<sha256>/...
  <output_dir>/metadata.csv
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


STATUS_COLUMNS = {
    "rendered",
    "voxelized",
    "cond_rendered",
}


def parse_indices(value: str) -> list[str]:
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(str(i) for i in range(int(start), int(end) + 1))
        else:
            indices.append(part)
    return indices


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def completion_score(row: pd.Series) -> int:
    score = 0
    for column, value in row.items():
        if column in STATUS_COLUMNS or column.startswith(("feature_", "latent_", "ss_latent_")):
            score += int(truthy(value))
        elif pd.notna(value):
            score += 1
    return score


def same_file(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def copy_file(src: Path, dst: Path, overwrite: str, dry_run: bool) -> str:
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


def copy_tree_contents(src_root: Path, dst_root: Path, overwrite: str, dry_run: bool) -> dict:
    stats = {
        "copied": 0,
        "skipped_same": 0,
        "skipped_conflict": 0,
        "conflict": 0,
    }
    if not src_root.exists():
        return stats

    files = [path for path in src_root.rglob("*") if path.is_file()]
    for src in tqdm(files, desc=f"Copying {src_root.name}", leave=False):
        rel = src.relative_to(src_root)
        status = copy_file(src, dst_root / rel, overwrite=overwrite, dry_run=dry_run)
        stats[status] += 1
    return stats


def load_metadata(metadata_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for path in metadata_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "sha256" not in df.columns:
            raise ValueError(f"metadata missing sha256 column: {path}")
        df["_source_metadata"] = path.name
        parts.append(df)

    if not parts:
        raise FileNotFoundError("No metadata CSV files found")

    merged = pd.concat(parts, ignore_index=True)
    duplicates = merged[merged.duplicated("sha256", keep=False)].copy()

    rows = []
    for _, group in merged.groupby("sha256", sort=False):
        if len(group) == 1:
            rows.append(group.iloc[0])
            continue
        scored = group.copy()
        scored["_completion_score"] = scored.apply(completion_score, axis=1)
        rows.append(scored.sort_values("_completion_score", ascending=False).iloc[0].drop(labels=["_completion_score"]))

    result = pd.DataFrame(rows)
    result = result.drop(columns=["_source_metadata"], errors="ignore")
    return result, duplicates


def discover_latent_models(latents_root: Path) -> list[str]:
    if not latents_root.exists():
        return []
    return sorted(path.name for path in latents_root.iterdir() if path.is_dir())


def refresh_metadata_status(metadata: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sha256"] = metadata["sha256"].astype(str)

    if "cond_rendered" not in metadata.columns:
        metadata["cond_rendered"] = False
    metadata["cond_rendered"] = metadata["sha256"].apply(
        lambda sha: (output_dir / "renders_cond" / sha / "transforms.json").exists()
    )

    latents_root = output_dir / "latents"
    for model in discover_latent_models(latents_root):
        column = f"latent_{model}"
        metadata[column] = metadata["sha256"].apply(
            lambda sha, model=model: (latents_root / model / f"{sha}.npz").exists()
        )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded FaceScape TRELLIS outputs")
    parser.add_argument("--source_root", type=Path, default=Path("/root/autodl-tmp"),
                        help="Directory containing preprocess_facescape_i shard directories")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Final TRELLIS training data root to create/update")
    parser.add_argument("--indices", type=str, default="1-5",
                        help="Shard indices, e.g. 1-5 or 1,2,3,4,5")
    parser.add_argument("--source_dir_pattern", type=str, default="preprocess_facescape_{i}",
                        help="Pattern for per-shard source directories under source_root. Use '.' for legacy flat source_root layout.")
    parser.add_argument("--latents_pattern", type=str, default="latents_{i}",
                        help="Pattern for sharded latent directories")
    parser.add_argument("--renders_cond_pattern", type=str, default="renders_cond_{i}",
                        help="Pattern for sharded condition directories")
    parser.add_argument("--metadata_pattern", type=str, default="metadata_{i}.csv",
                        help="Pattern for sharded metadata CSV files")
    parser.add_argument("--overwrite", choices=["skip", "replace", "error"], default="skip",
                        help="How to handle existing files with different sizes")
    parser.add_argument("--dry_run", action="store_true", help="Report actions without copying or writing")
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copy_totals = {}
    metadata_paths = []
    for index in indices:
        if args.source_dir_pattern == ".":
            shard_root = args.source_root
        else:
            shard_root = args.source_root / args.source_dir_pattern.format(i=index)
        latents_src = shard_root / args.latents_pattern.format(i=index)
        renders_cond_src = shard_root / args.renders_cond_pattern.format(i=index)
        metadata_path = shard_root / args.metadata_pattern.format(i=index)

        if metadata_path.exists():
            metadata_paths.append(metadata_path)
        else:
            print(f"[WARN] metadata not found: {metadata_path}")

        for label, src, dst in [
            ("latents", latents_src, args.output_dir / "latents"),
            ("renders_cond", renders_cond_src, args.output_dir / "renders_cond"),
        ]:
            if not src.exists():
                print(f"[WARN] {label} source not found: {src}")
                continue
            stats = copy_tree_contents(src, dst, overwrite=args.overwrite, dry_run=args.dry_run)
            copy_totals[f"{label}_{index}"] = stats

    metadata, duplicates = load_metadata(metadata_paths)
    metadata = refresh_metadata_status(metadata, args.output_dir)

    if not args.dry_run:
        metadata.to_csv(args.output_dir / "metadata.csv", index=False)
        if not duplicates.empty:
            duplicates.to_csv(args.output_dir / "metadata_duplicates.csv", index=False)

    print("[SUMMARY] copied files:")
    for key, stats in copy_totals.items():
        print(f"  - {key}: {stats}")
    print(f"[SUMMARY] metadata rows: {len(metadata)}")
    print(f"[SUMMARY] cond_rendered: {int(metadata['cond_rendered'].sum())}")
    latent_columns = [column for column in metadata.columns if column.startswith("latent_")]
    for column in latent_columns:
        print(f"[SUMMARY] {column}: {int(metadata[column].sum())}")
    if not duplicates.empty:
        print(f"[SUMMARY] duplicate metadata rows written to metadata_duplicates.csv: {len(duplicates)}")
    if args.dry_run:
        print("[SUMMARY] dry run: no files were copied and metadata.csv was not written")


if __name__ == "__main__":
    main()
