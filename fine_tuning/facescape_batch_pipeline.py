import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(args, timeout=None):
    print("\n[RUN] " + " ".join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=PROJECT_ROOT, check=True, timeout=timeout)


def build_metadata(dataset_root: Path, output_dir: Path):
    run_command([
        sys.executable,
        "dataset_toolkits/build_metadata.py",
        "FaceScape",
        "--dataset_root",
        dataset_root,
        "--output_dir",
        output_dir,
    ])


def read_metadata(output_dir: Path) -> pd.DataFrame:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    metadata["sha256"] = metadata["sha256"].astype(str)
    return metadata


def read_failed(output_dir: Path) -> set:
    failed_path = output_dir / "failed_batches.csv"
    if not failed_path.exists():
        return set()
    failed = pd.read_csv(failed_path)
    if "sha256" not in failed.columns:
        return set()
    return set(failed["sha256"].astype(str).tolist())


def remove_failed(output_dir: Path, sha256s: list):
    if not sha256s:
        return
    failed_path = output_dir / "failed_batches.csv"
    columns = ["sha256", "batch_index", "stage", "reason", "failed_at"]
    if not failed_path.exists():
        return
    failed = pd.read_csv(failed_path)
    if "sha256" not in failed.columns:
        return
    before = len(failed)
    failed = failed[~failed["sha256"].astype(str).isin(set(sha256s))]
    if failed.empty:
        failed = pd.DataFrame(columns=[column for column in columns if column in failed.columns] or columns)
    failed.to_csv(failed_path, index=False)
    removed = before - len(failed)
    if removed:
        print(f"[INFO] Removed {removed} recovered samples from {failed_path}", flush=True)


def append_failed(output_dir: Path, sha256s: list, batch_index: int, stage: str, reason: str):
    failed_path = output_dir / "failed_batches.csv"
    records = pd.DataFrame.from_records([
        {
            "sha256": sha256,
            "batch_index": batch_index,
            "stage": stage,
            "reason": reason,
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for sha256 in sha256s
    ])
    if failed_path.exists():
        old = pd.read_csv(failed_path)
        records = pd.concat([old, records], ignore_index=True)
        records = records.drop_duplicates(subset=["sha256"], keep="last")
    records.to_csv(failed_path, index=False)


def feature_path(output_dir: Path, model: str, sha256: str) -> Path:
    return output_dir / "features" / model / f"{sha256}.npz"


def is_feature_done(output_dir: Path, model: str, sha256: str) -> bool:
    path = feature_path(output_dir, model, sha256)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return "indices" in data and "patchtokens" in data
    except Exception:
        return False


def cleanup_completed_intermediates(output_dir: Path, model: str) -> int:
    metadata = read_metadata(output_dir)
    completed = [
        sha256
        for sha256 in metadata["sha256"].tolist()
        if is_feature_done(output_dir, model, sha256)
    ]
    before = count_intermediates(output_dir, completed)
    cleanup_intermediates(output_dir, completed)
    return before


def count_intermediates(output_dir: Path, sha256s: list) -> int:
    count = 0
    for sha256 in sha256s:
        if (output_dir / "renders" / sha256).exists():
            count += 1
        if (output_dir / "voxels" / f"{sha256}.ply").exists():
            count += 1
    return count


def is_truthy_status(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def filter_metadata_success(output_dir: Path, sha256s: list, field: str) -> tuple[list, list]:
    metadata = read_metadata(output_dir).set_index("sha256")
    success = []
    failed = []
    for sha256 in sha256s:
        if sha256 in metadata.index and is_truthy_status(metadata.loc[sha256, field]):
            success.append(sha256)
        else:
            failed.append(sha256)
    return success, failed


def split_feature_success(output_dir: Path, model: str, sha256s: list) -> tuple[list, list]:
    success = [sha256 for sha256 in sha256s if is_feature_done(output_dir, model, sha256)]
    failed = [sha256 for sha256 in sha256s if sha256 not in success]
    return success, failed


def is_failed_metadata_row(row, feature_field: str) -> bool:
    return (
        not is_truthy_status(row.get("rendered", False))
        or not is_truthy_status(row.get("voxelized", False))
        or not is_truthy_status(row.get(feature_field, False))
    )


def read_metadata_failed(output_dir: Path, model: str) -> set:
    metadata = read_metadata(output_dir)
    feature_field = f"feature_{model}"
    return set(
        row["sha256"]
        for _, row in metadata.iterrows()
        if is_failed_metadata_row(row, feature_field)
    )


def select_pending_batch(
    output_dir: Path,
    model: str,
    batch_size: int,
    continue_on_failed: bool,
    retry_failed: bool,
) -> list:
    metadata = read_metadata(output_dir)
    recorded_failed = read_failed(output_dir) if continue_on_failed and not retry_failed else set()
    metadata_failed = read_metadata_failed(output_dir, model) if retry_failed else None

    pending = []
    for sha256 in metadata["sha256"].tolist():
        if metadata_failed is not None and sha256 not in metadata_failed:
            continue
        if sha256 in recorded_failed:
            continue
        if not is_feature_done(output_dir, model, sha256):
            pending.append(sha256)
        if len(pending) >= batch_size:
            break
    if retry_failed:
        print(
            f"[INFO] retry_failed=True; metadata failed candidates={len(metadata_failed or set())}, "
            f"selected={len(pending)}",
            flush=True,
        )
    return pending


def write_instances(output_dir: Path, batch_index: int, sha256s: list) -> Path:
    batch_dir = output_dir / "batch_instances"
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"batch_{batch_index:06d}.txt"
    path.write_text("\n".join(sha256s) + "\n", encoding="utf-8")
    return path


def run_render(opt, instances_path: Path):
    args = [
        sys.executable,
        "fine_tuning/facescape_render.py",
        "--dataset_root",
        opt.dataset_root,
        "--output_dir",
        opt.output_dir,
        "--instances",
        instances_path,
        "--num_views",
        opt.num_views,
        "--max_workers",
        opt.render_workers,
        "--blender_batch_size",
        opt.blender_batch_size,
        "--timeout",
        opt.render_timeout,
    ]
    if not opt.enable_denoise:
        args.append("--profile_disable_denoise")
    run_command(args)


def run_voxelize(opt, instances_path: Path):
    run_command([
        sys.executable,
        "fine_tuning/voxelize.py",
        "FaceScape",
        "--output_dir",
        opt.output_dir,
        "--instances",
        instances_path,
        "--max_workers",
        opt.voxel_workers,
        "--timeout",
        opt.voxel_timeout,
    ])


def run_extract_feature(opt, instances_path: Path):
    run_command([
        sys.executable,
        "fine_tuning/facescape_extract_feature.py",
        "--output_dir",
        opt.output_dir,
        "--instances",
        instances_path,
        "--model",
        opt.model,
        "--batch_size",
        opt.feature_batch_size,
        "--overwrite",
    ], timeout=opt.feature_timeout)


def cleanup_intermediates(output_dir: Path, sha256s: list):
    for sha256 in sha256s:
        render_dir = output_dir / "renders" / sha256
        voxel_path = output_dir / "voxels" / f"{sha256}.ply"
        if render_dir.exists():
            shutil.rmtree(render_dir)
            print(f"[CLEAN] removed {render_dir}", flush=True)
        if voxel_path.exists():
            voxel_path.unlink()
            print(f"[CLEAN] removed {voxel_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FaceScape preprocessing in small batches to limit disk usage."
    )
    parser.add_argument("machine_index", type=int,
                        help="Output suffix i. Results are stored in /root/autodl-tmp/preprocess_facescape_{i} by default.")
    parser.add_argument("--dataset_root", type=Path, default=Path("/root/autodl-tmp/facescape"))
    parser.add_argument("--work_root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Override output dir. Default: <work_root>/preprocess_facescape_<machine_index>")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=150)
    parser.add_argument("--render_workers", type=int, default=8)
    parser.add_argument("--blender_batch_size", type=int, default=1)
    parser.add_argument("--render_timeout", type=float, default=300.0,
                        help="Per-sample Blender render timeout in seconds.")
    parser.add_argument("--voxel_workers", type=int, default=8)
    parser.add_argument("--voxel_timeout", type=float, default=10.0)
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--feature_batch_size", type=int, default=24)
    parser.add_argument("--feature_timeout", type=float, default=600.0,
                        help="Whole-batch feature extraction timeout in seconds.")
    parser.add_argument("--enable_denoise", action="store_true",
                        help="Keep Cycles denoising enabled. Default disables denoise for faster validation preprocessing.")
    parser.add_argument("--continue_on_failed", action="store_true",
                        help="Record failed samples and continue with later batches instead of stopping.")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Only retry samples whose metadata status is incomplete, including samples recorded in failed_batches.csv.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional limit for smoke tests.")
    return parser.parse_args()


def handle_failed(opt, sha256s: list, batch_index: int, stage: str, reason: str):
    if not sha256s:
        return
    print(f"[ERROR] Batch {batch_index} has {len(sha256s)} failed samples at {stage}: {sha256s}", flush=True)
    print(f"[ERROR] Reason: {reason}", flush=True)
    if opt.continue_on_failed:
        append_failed(opt.output_dir, sha256s, batch_index, stage, reason)
        cleanup_intermediates(opt.output_dir, sha256s)
        return
    raise RuntimeError(
        f"Some samples failed at {stage}. "
        "Intermediates for failed samples are kept for inspection. "
        "Use --continue_on_failed to skip them and continue."
    )


def main():
    opt = parse_args()
    if opt.output_dir is None:
        opt.output_dir = opt.work_root / f"preprocess_facescape_{opt.machine_index}"

    opt.dataset_root = opt.dataset_root.resolve()
    opt.output_dir = opt.output_dir.resolve()
    opt.work_root = opt.work_root.resolve()

    if not opt.dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {opt.dataset_root}")
    opt.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset_root={opt.dataset_root}", flush=True)
    print(f"[INFO] output_dir={opt.output_dir}", flush=True)
    print(f"[INFO] batch_size={opt.batch_size}", flush=True)
    print(f"[INFO] retry_failed={opt.retry_failed}", flush=True)

    build_metadata(opt.dataset_root, opt.output_dir)
    cleaned = cleanup_completed_intermediates(opt.output_dir, opt.model)
    if cleaned:
        print(f"[CLEAN] removed {cleaned} stale intermediate paths for completed samples", flush=True)

    batch_index = 0
    while True:
        if opt.max_batches is not None and batch_index >= opt.max_batches:
            print(f"[DONE] Reached max_batches={opt.max_batches}", flush=True)
            break

        cleaned = cleanup_completed_intermediates(opt.output_dir, opt.model)
        if cleaned:
            print(f"[CLEAN] removed {cleaned} stale intermediate paths for completed samples", flush=True)

        sha256s = select_pending_batch(
            opt.output_dir,
            opt.model,
            opt.batch_size,
            opt.continue_on_failed,
            opt.retry_failed,
        )
        if not sha256s:
            print("[DONE] No pending samples left.", flush=True)
            break

        print(f"\n[BATCH {batch_index}] selected {len(sha256s)} samples", flush=True)
        instances_path = write_instances(opt.output_dir, batch_index, sha256s)

        run_render(opt, instances_path)
        build_metadata(opt.dataset_root, opt.output_dir)
        sha256s, failed = filter_metadata_success(opt.output_dir, sha256s, "rendered")
        handle_failed(opt, failed, batch_index, "render", "rendered metadata is not True; check merged_records/rendered_*.csv")
        if not sha256s:
            batch_index += 1
            continue
        instances_path = write_instances(opt.output_dir, batch_index, sha256s)

        run_voxelize(opt, instances_path)
        build_metadata(opt.dataset_root, opt.output_dir)
        sha256s, failed = filter_metadata_success(opt.output_dir, sha256s, "voxelized")
        handle_failed(opt, failed, batch_index, "voxelize", "voxelized metadata is not True; check merged_records/voxelized_*.csv and skipped_voxels_*.json")
        if not sha256s:
            batch_index += 1
            continue
        instances_path = write_instances(opt.output_dir, batch_index, sha256s)

        feature_failure_reason = "feature npz not produced"
        try:
            run_extract_feature(opt, instances_path)
        except subprocess.TimeoutExpired:
            feature_failure_reason = f"feature extraction timed out after {opt.feature_timeout} seconds"
            print(f"[ERROR] {feature_failure_reason}.", flush=True)
        except subprocess.CalledProcessError as e:
            feature_failure_reason = f"feature extraction exited with code {e.returncode}"
            print(f"[ERROR] {feature_failure_reason}.", flush=True)
        build_metadata(opt.dataset_root, opt.output_dir)

        completed, failed = split_feature_success(opt.output_dir, opt.model, sha256s)

        if completed:
            build_metadata(opt.dataset_root, opt.output_dir)
            remove_failed(opt.output_dir, completed)

        cleanup_intermediates(opt.output_dir, completed)

        handle_failed(opt, failed, batch_index, "feature", feature_failure_reason)

        batch_index += 1

    build_metadata(opt.dataset_root, opt.output_dir)


if __name__ == "__main__":
    main()
