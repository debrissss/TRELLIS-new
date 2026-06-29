"""
Scan or clean empty-alpha frames in TRELLIS renders_cond data.

By default this script is scan-only. With --clean, it removes frames whose RGBA
alpha channel has no foreground pixels from renders_cond/<sha>/transforms.json.
If a sample has no valid frames left, it is removed from metadata.csv.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def image_has_foreground(path: Path, alpha_threshold: int) -> bool:
    image = Image.open(path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = np.array(image.getchannel(3))
    return bool((alpha > alpha_threshold).any())


def load_metadata(dataset_dir: Path, include_all_metadata_rows: bool) -> pd.DataFrame:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"metadata.csv must contain sha256 column: {metadata_path}")
    metadata["sha256"] = metadata["sha256"].astype(str)

    if not include_all_metadata_rows and "cond_rendered" in metadata.columns:
        metadata = metadata[metadata["cond_rendered"].apply(truthy)].copy()
    return metadata


def scan_sample(dataset_dir: Path, sha256: str, alpha_threshold: int) -> dict:
    render_dir = dataset_dir / "renders_cond" / sha256
    transforms_path = render_dir / "transforms.json"
    result = {
        "sha256": sha256,
        "transforms_path": str(transforms_path),
        "total_frames": 0,
        "valid_frames": [],
        "bad_frames": [],
        "missing_frames": [],
        "error": "",
        "metadata": None,
    }

    if not transforms_path.exists():
        result["error"] = "missing_transforms"
        return result

    try:
        with open(transforms_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        result["error"] = f"invalid_json: {e}"
        return result

    frames = metadata.get("frames")
    if not isinstance(frames, list):
        result["error"] = "missing_or_invalid_frames"
        return result

    result["metadata"] = metadata
    result["total_frames"] = len(frames)

    for index, frame in enumerate(frames):
        file_path = frame.get("file_path") if isinstance(frame, dict) else None
        if not file_path:
            result["bad_frames"].append({"index": index, "file_path": "", "reason": "missing_file_path"})
            continue

        image_path = render_dir / file_path
        if not image_path.exists():
            result["missing_frames"].append({"index": index, "file_path": file_path, "reason": "missing_file"})
            continue

        try:
            has_foreground = image_has_foreground(image_path, alpha_threshold)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            result["bad_frames"].append({"index": index, "file_path": file_path, "reason": f"image_error: {e}"})
            continue

        if has_foreground:
            result["valid_frames"].append(frame)
        else:
            result["bad_frames"].append({"index": index, "file_path": file_path, "reason": "empty_alpha"})

    return result


def write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def backup_file(path: Path, backup_suffix: str) -> Path:
    backup_path = path.with_name(f"{path.name}.{backup_suffix}")
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    shutil.copy2(path, backup_path)
    return backup_path


def iter_scan_results(dataset_dir: Path, sha256s: list[str], alpha_threshold: int, workers: int):
    if workers <= 1:
        for sha256 in tqdm(sha256s, desc="Scanning renders_cond"):
            yield scan_sample(dataset_dir, sha256, alpha_threshold)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda sha: scan_sample(dataset_dir, sha, alpha_threshold),
            sha256s,
        )
        yield from tqdm(results, total=len(sha256s), desc=f"Scanning renders_cond ({workers} workers)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan or clean empty-alpha frames from renders_cond transforms.json files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset root containing metadata.csv and renders_cond/<sha>/transforms.json.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Modify transforms.json and metadata.csv. Without this flag the script only scans and writes reports.",
    )
    parser.add_argument(
        "--alpha_threshold",
        type=int,
        default=0,
        help="A frame is valid if any alpha value is greater than this threshold. Defaults to 0.",
    )
    parser.add_argument(
        "--include_all_metadata_rows",
        action="store_true",
        help="Scan all metadata rows instead of only cond_rendered=true rows when the column exists.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create backups in --clean mode.",
    )
    parser.add_argument(
        "--report_prefix",
        type=Path,
        default=None,
        help="Report prefix. Defaults to <dataset_dir>/empty_alpha_scan_<timestamp>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of parallel scanner threads. Use 1 for sequential scan. Defaults to min(16, CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.alpha_threshold < 0 or args.alpha_threshold > 255:
        raise ValueError("--alpha_threshold must be in [0, 255]")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = args.report_prefix or (dataset_dir / f"empty_alpha_scan_{timestamp}")
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(dataset_dir, args.include_all_metadata_rows)
    sha256s = metadata["sha256"].dropna().astype(str).tolist()

    frame_rows = []
    sample_rows = []
    remove_sha256s = set()
    changed_transforms = []

    for result in iter_scan_results(dataset_dir, sha256s, args.alpha_threshold, args.workers):
        sha256 = result["sha256"]
        bad_count = len(result["bad_frames"])
        missing_count = len(result["missing_frames"])
        valid_count = len(result["valid_frames"])
        total_frames = result["total_frames"]
        error = result["error"]

        should_remove_sample = bool(error) or valid_count == 0
        if should_remove_sample:
            remove_sha256s.add(sha256)

        if bad_count or missing_count or should_remove_sample:
            sample_rows.append({
                "sha256": sha256,
                "total_frames": total_frames,
                "valid_frames": valid_count,
                "bad_frames": bad_count,
                "missing_frames": missing_count,
                "remove_sample": should_remove_sample,
                "error": error,
            })

        for item in result["bad_frames"]:
            frame_rows.append({"sha256": sha256, **item})
        for item in result["missing_frames"]:
            frame_rows.append({"sha256": sha256, **item})

        if args.clean and not error and bad_count:
            transforms_path = Path(result["transforms_path"])
            cleaned = dict(result["metadata"])
            cleaned["frames"] = result["valid_frames"]
            if not args.no_backup:
                backup_file(transforms_path, f"before_empty_alpha_clean_{timestamp}")
            write_json(transforms_path, cleaned)
            changed_transforms.append(str(transforms_path))

    sample_report = report_prefix.with_suffix(".samples.csv")
    frame_report = report_prefix.with_suffix(".frames.csv")
    pd.DataFrame(sample_rows).to_csv(sample_report, index=False)
    pd.DataFrame(frame_rows).to_csv(frame_report, index=False)

    print(f"[SUMMARY] scanned samples: {len(sha256s)}")
    print(f"[SUMMARY] samples with issues: {len(sample_rows)}")
    print(f"[SUMMARY] bad or missing frames: {len(frame_rows)}")
    print(f"[SUMMARY] samples to remove from metadata: {len(remove_sha256s)}")
    print(f"[SUMMARY] sample report: {sample_report}")
    print(f"[SUMMARY] frame report: {frame_report}")

    if not args.clean:
        print("[SUMMARY] scan-only mode: no files were modified. Re-run with --clean to apply changes.")
        return

    metadata_path = dataset_dir / "metadata.csv"
    full_metadata = pd.read_csv(metadata_path)
    full_metadata["sha256"] = full_metadata["sha256"].astype(str)
    cleaned_metadata = full_metadata[~full_metadata["sha256"].isin(remove_sha256s)].copy()

    if not args.no_backup:
        backup = backup_file(metadata_path, f"before_empty_alpha_clean_{timestamp}")
        print(f"[INFO] metadata backup written to {backup}")

    cleaned_metadata.to_csv(metadata_path, index=False)
    print(f"[SUMMARY] cleaned transforms.json files: {len(changed_transforms)}")
    print(f"[SUMMARY] metadata rows before: {len(full_metadata)}")
    print(f"[SUMMARY] metadata rows after: {len(cleaned_metadata)}")


if __name__ == "__main__":
    main()
