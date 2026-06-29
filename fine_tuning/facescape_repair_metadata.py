import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_TOOLKITS_DIR = PROJECT_ROOT / "dataset_toolkits"
sys.path.insert(0, str(DATASET_TOOLKITS_DIR))

from utils import get_file_hash  # noqa: E402


STATUS_FIELDS = {
    "rendered",
    "voxelized",
    "num_voxels",
    "cond_rendered",
}

RECORD_PREFIXES = (
    "downloaded_",
    "rendered_",
    "aesthetic_scores_",
    "voxelized_",
    "cond_rendered_",
    "feature_",
    "latent_",
    "ss_latent_",
)


def run_command(args):
    print("[RUN] " + " ".join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=PROJECT_ROOT, check=True)


def is_status_col(column: str) -> bool:
    return (
        column in STATUS_FIELDS
        or column.startswith("feature_")
        or column.startswith("latent_")
        or column.startswith("ss_latent_")
    )


def default_value_for_status(column: str):
    if column == "num_voxels":
        return 0
    return False


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if not isinstance(value, (bool, int, float))
        else bool(value)
    )


def aggregate_duplicate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sha256"] = metadata["sha256"].astype(str)
    status_cols = [column for column in metadata.columns if is_status_col(column)]
    base_cols = [column for column in metadata.columns if column not in status_cols]

    base = metadata[base_cols].drop_duplicates("sha256", keep="first").set_index("sha256")
    grouped = metadata.set_index("sha256")

    for column in status_cols:
        if column == "num_voxels":
            values = pd.to_numeric(grouped[column], errors="coerce").fillna(0).groupby(level=0).max()
        else:
            values = coerce_bool_series(grouped[column]).groupby(level=0).max()
        base[column] = values.reindex(base.index).fillna(default_value_for_status(column))

    return base.reset_index()


def sanitize_record_csvs(output_dir: Path, timestamp: str):
    for csv_path in sorted(output_dir.glob("*.csv")):
        if not csv_path.name.endswith(".csv"):
            continue
        if not csv_path.name.startswith(RECORD_PREFIXES):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read record csv {csv_path}: {e}", flush=True)
            continue
        if "sha256" not in df.columns or not df["sha256"].duplicated().any():
            continue

        duplicate_report = output_dir / f"{csv_path.stem}.duplicate_sha_{timestamp}.csv"
        df[df["sha256"].astype(str).duplicated(keep=False)].to_csv(duplicate_report, index=False)

        df["sha256"] = df["sha256"].astype(str)
        status_cols = [column for column in df.columns if is_status_col(column)]
        base_cols = [column for column in df.columns if column not in status_cols]
        base = df[base_cols].drop_duplicates("sha256", keep="last").set_index("sha256")
        grouped = df.set_index("sha256")

        for column in status_cols:
            if column == "num_voxels":
                values = pd.to_numeric(grouped[column], errors="coerce").fillna(0).groupby(level=0).max()
            else:
                values = coerce_bool_series(grouped[column]).groupby(level=0).max()
            base[column] = values.reindex(base.index).fillna(default_value_for_status(column))

        backup = output_dir / f"{csv_path.name}.before_dedup_{timestamp}"
        shutil.copy2(csv_path, backup)
        base.reset_index().to_csv(csv_path, index=False)
        print(f"[WARN] Deduplicated record csv {csv_path}; duplicate rows saved to {duplicate_report}", flush=True)


def write_duplicate_physical_paths_report(metadata: pd.DataFrame, dataset_root: Path, output_dir: Path, timestamp: str, prefix: str):
    duplicates = metadata[metadata["sha256"].astype(str).duplicated(keep=False)].copy()
    if len(duplicates) == 0:
        return
    duplicates["sha256"] = duplicates["sha256"].astype(str)
    duplicates = duplicates.sort_values(["sha256", "local_path"])
    duplicates["physical_path"] = duplicates["local_path"].map(lambda path: str(dataset_root / str(path)))
    report = duplicates[["sha256", "physical_path"]]
    report_path = output_dir / f"{prefix}.duplicate_physical_paths_{timestamp}.csv"
    report.to_csv(report_path, index=False)
    print(f"[WARN] Duplicate physical path report written to {report_path}", flush=True)


def write_all_ply_duplicate_report(dataset_root: Path, output_dir: Path, timestamp: str):
    records = []
    ply_paths = sorted(dataset_root.rglob("*.ply"))
    print(f"[INFO] Scanning all PLY files for duplicate content hashes: {len(ply_paths)} files", flush=True)
    for path in ply_paths:
        try:
            records.append({
                "sha256": get_file_hash(str(path)),
                "physical_path": str(path),
            })
        except Exception as e:
            print(f"[WARN] Failed to hash {path}: {e}", flush=True)

    if not records:
        return

    all_ply = pd.DataFrame.from_records(records)
    duplicate_rows = all_ply[all_ply["sha256"].duplicated(keep=False)].copy()
    if len(duplicate_rows) == 0:
        print("[INFO] No duplicate content hashes found among all PLY files.", flush=True)
        return

    duplicate_rows = duplicate_rows.sort_values(["sha256", "physical_path"])
    report_path = output_dir / f"all_ply.duplicate_physical_paths_{timestamp}.csv"
    duplicate_rows[["sha256", "physical_path"]].to_csv(report_path, index=False)
    print(f"[WARN] All-PLY duplicate physical path report written to {report_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-scan FaceScape metadata and preserve existing preprocessing progress by sha256."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--temp_dir", type=Path, default=None,
                        help="Temporary output dir for the re-scanned metadata. Defaults to <output_dir>_rescan_meta.")
    parser.add_argument("--keep_temp", action="store_true")
    return parser.parse_args()


def main():
    opt = parse_args()
    dataset_root = opt.dataset_root.resolve()
    output_dir = opt.output_dir.resolve()
    temp_dir = opt.temp_dir.resolve() if opt.temp_dir is not None else Path(f"{output_dir}_rescan_meta")

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"metadata.before_rescan_{timestamp}.csv"
    shutil.copy2(metadata_path, backup_path)
    print(f"[INFO] Backed up old metadata to {backup_path}", flush=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        sys.executable,
        "dataset_toolkits/build_metadata.py",
        "FaceScape",
        "--dataset_root",
        dataset_root,
        "--output_dir",
        temp_dir,
    ])

    old = pd.read_csv(metadata_path)
    new = pd.read_csv(temp_dir / "metadata.csv")
    old["sha256"] = old["sha256"].astype(str)
    new["sha256"] = new["sha256"].astype(str)

    old_dups = old[old["sha256"].duplicated(keep=False)]
    if len(old_dups) > 0:
        old_dups_path = output_dir / f"metadata.old_duplicate_sha_{timestamp}.csv"
        old_dups.to_csv(old_dups_path, index=False)
        print(f"[WARN] Old metadata has {len(old_dups)} duplicate sha rows: {old_dups_path}", flush=True)
        if "local_path" in old_dups.columns:
            write_duplicate_physical_paths_report(old, dataset_root, output_dir, timestamp, "metadata.old")

    new_dups = new[new["sha256"].duplicated(keep=False)]
    if len(new_dups) > 0:
        new_dups_path = output_dir / f"metadata.rescan_duplicate_sha_{timestamp}.csv"
        new_dups.to_csv(new_dups_path, index=False)
        print(f"[WARN] Rescanned metadata has {len(new_dups)} duplicate sha rows: {new_dups_path}", flush=True)
        if "local_path" in new_dups.columns:
            write_duplicate_physical_paths_report(new, dataset_root, output_dir, timestamp, "metadata.rescan")

    write_all_ply_duplicate_report(dataset_root, output_dir, timestamp)

    old = aggregate_duplicate_metadata(old).set_index("sha256")
    new = new.drop_duplicates("sha256", keep="first").set_index("sha256")

    status_cols = [column for column in old.columns if is_status_col(column)]
    common = new.index.intersection(old.index)
    for column in status_cols:
        if column not in new.columns:
            new[column] = default_value_for_status(column)
        new.loc[common, column] = old.loc[common, column]

    repaired = new.reset_index()
    repaired.to_csv(metadata_path, index=False)
    print(f"[INFO] Repaired metadata written to {metadata_path}", flush=True)
    print(f"[INFO] Old unique assets: {len(old)}", flush=True)
    print(f"[INFO] Rescanned unique assets: {len(new)}", flush=True)
    print(f"[INFO] Preserved status columns: {status_cols}", flush=True)

    sanitize_record_csvs(output_dir, timestamp)

    run_command([
        sys.executable,
        "dataset_toolkits/build_metadata.py",
        "FaceScape",
        "--dataset_root",
        dataset_root,
        "--output_dir",
        output_dir,
    ])

    if not opt.keep_temp:
        shutil.rmtree(temp_dir)
        print(f"[INFO] Removed temporary metadata dir {temp_dir}", flush=True)


if __name__ == "__main__":
    main()
