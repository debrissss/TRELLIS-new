import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add an aesthetic_score column to metadata.csv for TRELLIS fine-tuning."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing metadata.csv.",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=5.0,
        help="Value assigned to aesthetic_score. Defaults to 5.0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite aesthetic_score if the column already exists.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create metadata.csv.before_aesthetic_score backup before writing.",
    )
    return parser.parse_args()


def main():
    opt = parse_args()
    metadata_path = opt.output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"metadata.csv must contain sha256 column: {metadata_path}")

    if "aesthetic_score" in metadata.columns and not opt.overwrite:
        print("[INFO] aesthetic_score already exists; no changes written. Use --overwrite to replace it.")
        return

    if opt.backup:
        backup_path = metadata_path.with_name("metadata.csv.before_aesthetic_score")
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        shutil.copy2(metadata_path, backup_path)
        print(f"[INFO] Backup written to {backup_path}")

    metadata["aesthetic_score"] = opt.score
    metadata.to_csv(metadata_path, index=False)
    print(f"[INFO] Wrote aesthetic_score={opt.score} for {len(metadata)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
