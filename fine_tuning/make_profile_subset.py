import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create profile_i.txt containing the first i sha256 values from metadata.csv."
    )
    parser.add_argument("count", type=int, help="Number of samples to write")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory containing metadata.csv")
    parser.add_argument("--profile_dir", type=Path, default=Path("/root/autodl-tmp"),
                        help="Directory to save profile_i.txt")
    parser.add_argument("--only_feature_done", action="store_true",
                        help="Only select rows with feature_<model> marked True")
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg",
                        help="Feature model name used with --only_feature_done")
    return parser.parse_args()


def main():
    opt = parse_args()
    if opt.count <= 0:
        raise ValueError("count must be positive")

    metadata_path = opt.output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"metadata.csv must contain sha256 column: {metadata_path}")
    metadata["sha256"] = metadata["sha256"].astype(str)

    if opt.only_feature_done:
        column = f"feature_{opt.model}"
        if column not in metadata.columns:
            raise ValueError(f"metadata.csv does not contain {column}")
        metadata = metadata[metadata[column] == True]

    sha256s = metadata["sha256"].dropna().head(opt.count).tolist()
    if len(sha256s) < opt.count:
        print(f"[WARN] Requested {opt.count} samples, but only found {len(sha256s)} matching rows.")

    opt.profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = opt.profile_dir / f"profile_{opt.count}.txt"
    profile_path.write_text("\n".join(sha256s) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {len(sha256s)} samples to {profile_path}")


if __name__ == "__main__":
    main()
