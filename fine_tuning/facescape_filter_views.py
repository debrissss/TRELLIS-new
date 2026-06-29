"""
FaceScape original-camera view filtering for conditional normal maps.

This stage writes filtered FaceScape original camera view indices to
view_filters/{sha256} without depending on TRELLIS renders/{sha256}/transforms.json.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from camera_view_filter import compute_face_center  # noqa: E402
from fine_tuning.preprocess_stage1 import is_view_valid  # noqa: E402
from fine_tuning.utils.facescape_utils import get_file_sha256, get_subject_paths  # noqa: E402
from fine_tuning.utils.logger import get_logger  # noqa: E402


logger = get_logger("facescape_filter_views")


def _parse_camera_params(camera_path: Path) -> dict:
    with open(camera_path, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    camera_params = {}
    for k, v in raw_params.items():
        if "_" not in k:
            continue
        parts = k.split("_", 1)
        if not parts[0].isdigit():
            continue
        cam_id = int(parts[0])
        field_name = parts[1]
        camera_params.setdefault(cam_id, {})[field_name] = v
    return camera_params


def _resolve_camera_path(dataset_root: Path, subject_id: str, expr_name: str, local_path: str) -> Path:
    mesh_local_path = Path(local_path)
    parts = list(mesh_local_path.parts)
    if "closed_shapes_meshlib" in parts:
        idx = parts.index("closed_shapes_meshlib")
        direct_camera_path = dataset_root / Path(*parts[:idx]) / "aligned_camera_params" / subject_id / expr_name / "params.json"
        if direct_camera_path.exists():
            return direct_camera_path

    _, aligned_params_dir = get_subject_paths(dataset_root, subject_id)
    camera_path = aligned_params_dir / expr_name / "params.json"
    if camera_path.exists():
        return camera_path

    alternatives = list(aligned_params_dir.glob(f"**/{expr_name}/params.json"))
    if alternatives:
        return alternatives[0]
    raise FileNotFoundError(f"未能找到相机参数: subject_id={subject_id}, expr_name={expr_name}")


def _subject_expr_from_local_path(local_path: str) -> Tuple[str, str]:
    path = Path(local_path)
    if path.suffix.lower() != ".ply":
        raise ValueError(f"local_path 不是 PLY 文件: {local_path}")
    return path.parent.name, path.stem


def _camera_angle_x_from_params(params: dict) -> float:
    width = params.get("width", 518)
    K = np.array(params.get("K", [[1000, 0, width / 2], [0, 1000, 518 / 2], [0, 0, 1]]), dtype=np.float64)
    f_x = K[0, 0]
    return float(2.0 * np.arctan(width / (2.0 * f_x)))


def filter_views_for_mesh(
    dataset_root: Path,
    output_dir: Path,
    local_path: str,
    sha256: str,
    center_mode: str,
    thresh_left_back: float,
    thresh_right_back: float,
    thresh_up: float,
    thresh_down: float,
) -> dict:
    subject_id, expr_name = _subject_expr_from_local_path(local_path)
    mesh_path = dataset_root / local_path
    if not mesh_path.exists():
        raise FileNotFoundError(f"原始 mesh 不存在: {mesh_path}")

    actual_sha256 = get_file_sha256(mesh_path)
    if actual_sha256 != sha256:
        raise ValueError(f"metadata sha256 与文件不一致: {sha256} != {actual_sha256}")

    camera_path = _resolve_camera_path(dataset_root, subject_id, expr_name, local_path)
    camera_params = _parse_camera_params(camera_path)
    face_center = compute_face_center(center_mode, mesh_path, camera_params)

    frames = []
    for cam_id, params in sorted(camera_params.items()):
        if "Rt" not in params:
            continue

        Rt = np.array(params["Rt"], dtype=np.float64)
        R = Rt[:, :3]
        t = Rt[:, 3]
        camera_center = -np.dot(R.T, t)

        if not is_view_valid(
            camera_center=camera_center,
            face_center=face_center,
            thresh_left_back=thresh_left_back,
            thresh_right_back=thresh_right_back,
            thresh_up=thresh_up,
            thresh_down=thresh_down,
        ):
            continue

        frames.append({
            "file_path": f"normal_{cam_id:03d}.png",
            "cam_id": cam_id,
            "camera_angle_x": _camera_angle_x_from_params(params),
        })

    output_folder = output_dir / "view_filters" / sha256
    output_folder.mkdir(parents=True, exist_ok=True)
    output_data = {
        "subject_id": subject_id,
        "expr_name": expr_name,
        "camera_params": str(camera_path.relative_to(dataset_root)) if camera_path.is_relative_to(dataset_root) else str(camera_path),
        "coordinate_space": "facescape_original",
        "frames": frames,
    }
    with open(output_folder / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    return {"sha256": sha256, "view_filtered": True, "num_views": len(frames)}


def _worker(args_tuple):
    try:
        return filter_views_for_mesh(*args_tuple)
    except Exception as e:
        return {"sha256": args_tuple[3], "view_filtered": False, "num_views": 0, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter FaceScape original camera views for conditional normal maps")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--center_mode", type=str, default="camera_intersection",
                        choices=["origin", "mesh_mean", "mesh_bbox", "camera_intersection"])
    parser.add_argument("--thresh_left_back", type=float, default=100.0)
    parser.add_argument("--thresh_right_back", type=float, default=100.0)
    parser.add_argument("--thresh_up", type=float, default=40.0)
    parser.add_argument("--thresh_down", type=float, default=40.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise ValueError(f"dataset_root not found: {dataset_root}")

    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"metadata.csv not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns or "local_path" not in metadata.columns:
        raise ValueError("metadata.csv must contain sha256 and local_path columns")
    metadata = metadata[metadata["sha256"].notna() & metadata["local_path"].notna()].copy()
    metadata["sha256"] = metadata["sha256"].astype(str)

    if args.instances is not None:
        if os.path.exists(args.instances):
            with open(args.instances, "r", encoding="utf-8") as f:
                instances = [line.strip() for line in f if line.strip()]
        else:
            instances = [item.strip() for item in args.instances.split(",") if item.strip()]
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]

    worker_count = args.max_workers or max(1, os.cpu_count() - 2)
    task_args = [
        (
            dataset_root,
            output_dir,
            row["local_path"],
            row["sha256"],
            args.center_mode,
            args.thresh_left_back,
            args.thresh_right_back,
            args.thresh_up,
            args.thresh_down,
        )
        for _, row in metadata.iterrows()
    ]

    records = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_worker, item) for item in task_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering FaceScape views"):
            records.append(future.result())

    pd.DataFrame.from_records(records).to_csv(output_dir / f"view_filtered_{args.rank}.csv", index=False)
    failed = [record for record in records if not record.get("view_filtered")]
    logger.info(f"视角过滤完成: success={len(records) - len(failed)}, failed={len(failed)}")
    if failed:
        logger.error(f"失败样本示例: {failed[:10]}")


if __name__ == "__main__":
    main()
