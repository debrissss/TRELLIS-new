"""
FaceScape TRELLIS multi-view renderer.

This script reads original FaceScape meshes, lets the TRELLIS Blender renderer
normalize them, saves the normalized mesh.ply, and renders the feature views.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from subprocess import DEVNULL, PIPE, STDOUT, Popen, TimeoutExpired

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_TOOLKITS_DIR = PROJECT_ROOT / "dataset_toolkits"
sys.path.insert(0, str(DATASET_TOOLKITS_DIR))

from utils import sphere_hammersley_sequence  # noqa: E402


BLENDER_LINK = "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}")


def _render_complete(output_folder: str, num_views: int) -> bool:
    transforms_path = os.path.join(output_folder, "transforms.json")
    mesh_path = os.path.join(output_folder, "mesh.ply")
    if not os.path.exists(transforms_path) or not os.path.exists(mesh_path):
        return False

    try:
        with open(transforms_path, "r", encoding="utf-8") as f:
            transforms = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    frames = transforms.get("frames", [])
    if len(frames) < num_views:
        return False

    expected = {f"{i:03d}.png" for i in range(num_views)}
    rendered = {frame.get("file_path") for frame in frames[:num_views]}
    if not expected.issubset(rendered):
        return False

    return all(os.path.exists(os.path.join(output_folder, name)) for name in expected)


def _build_views(num_views: int) -> list:
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    return [
        {"yaw": y, "pitch": p, "radius": r, "fov": f}
        for y, p, r, f in zip(yaws, pitchs, radius, fov)
    ]


def _render(
    file_path: str,
    sha256: str,
    output_dir: str,
    num_views: int,
    verbose_blender: bool = False,
    profile_blender: bool = False,
    blender_log: bool = False,
    profile_disable_denoise: bool = False,
    profile_no_write: bool = False,
    skip_normalize: bool = False,
    timeout: float = 300.0,
) -> dict:
    output_folder = os.path.join(output_dir, "renders", sha256)

    if not os.path.exists(file_path):
        return {
            "sha256": sha256,
            "rendered": False,
            "error": f"source mesh not found: {file_path}",
        }

    views = _build_views(num_views)
    args = [
        BLENDER_PATH,
        "-b",
        "-P",
        str(DATASET_TOOLKITS_DIR / "blender_script" / "render.py"),
        "--",
        "--views",
        json.dumps(views),
        "--object",
        os.path.expanduser(file_path),
        "--resolution",
        "512",
        "--output_folder",
        output_folder,
        "--engine",
        "CYCLES",
        "--save_mesh",
    ]
    if skip_normalize:
        args.append("--skip_normalize")
    if profile_blender:
        args.append("--profile")
    if profile_disable_denoise:
        args.append("--profile_disable_denoise")
    if profile_no_write:
        args.append("--profile_no_write")

    start = time.perf_counter()
    proc = None
    try:
        if blender_log:
            proc = Popen(args)
            return_code = proc.wait(timeout=timeout)
        elif verbose_blender:
            proc = Popen(args, stdout=PIPE, stderr=STDOUT, text=True)
            output, _ = proc.communicate(timeout=timeout)
            for line in output.splitlines(True):
                if line.startswith("[INFO]") or line.startswith("[WARN]") or line.startswith("[PROFILE]"):
                    print(line, end="")
            return_code = proc.returncode
        else:
            proc = Popen(args, stdout=DEVNULL, stderr=DEVNULL)
            return_code = proc.wait(timeout=timeout)
    except TimeoutExpired:
        if proc is not None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except TimeoutExpired:
                pass
        wall_seconds = time.perf_counter() - start
        return {
            "sha256": sha256,
            "rendered": False,
            "wall_seconds": wall_seconds,
            "error": f"blender render timed out after {timeout} seconds",
        }
    wall_seconds = time.perf_counter() - start
    if return_code == 0 and _render_complete(output_folder, num_views):
        return {"sha256": sha256, "rendered": True, "wall_seconds": wall_seconds}

    return {
        "sha256": sha256,
        "rendered": False,
        "wall_seconds": wall_seconds,
        "error": f"blender render failed with exit code {return_code}",
    }


def _run_blender_process(args, verbose_blender: bool, blender_log: bool, timeout: float) -> int:
    proc = None
    try:
        if blender_log:
            proc = Popen(args)
            return proc.wait(timeout=timeout)
        if verbose_blender:
            proc = Popen(args, stdout=PIPE, stderr=STDOUT, text=True)
            output, _ = proc.communicate(timeout=timeout)
            for line in output.splitlines(True):
                if line.startswith("[INFO]") or line.startswith("[WARN]") or line.startswith("[PROFILE]") or line.startswith("[BATCH]"):
                    print(line, end="")
            return proc.returncode
        proc = Popen(args, stdout=DEVNULL, stderr=DEVNULL)
        return proc.wait(timeout=timeout)
    except TimeoutExpired:
        if proc is not None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except TimeoutExpired:
                pass
        return -9


def _render_batch(
    jobs: list,
    output_dir: str,
    num_views: int,
    verbose_blender: bool = False,
    profile_blender: bool = False,
    blender_log: bool = False,
    profile_disable_denoise: bool = False,
    profile_no_write: bool = False,
    skip_normalize: bool = False,
    timeout: float = 300.0,
) -> list:
    if not jobs:
        return []

    for job in jobs:
        if not os.path.exists(job["object"]):
            return [{
                "sha256": job["sha256"],
                "rendered": False,
                "error": f"source mesh not found: {job['object']}",
            }]

    with tempfile.TemporaryDirectory(prefix="facescape_render_batch_") as tmp_dir:
        jobs_path = os.path.join(tmp_dir, "jobs.json")
        records_path = os.path.join(tmp_dir, "records.json")
        with open(jobs_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f)

        args = [
            BLENDER_PATH,
            "-b",
            "-P",
            str(DATASET_TOOLKITS_DIR / "blender_script" / "render_batch.py"),
            "--",
            "--jobs",
            jobs_path,
            "--records",
            records_path,
            "--resolution",
            "512",
            "--engine",
            "CYCLES",
            "--save_mesh",
        ]
        if profile_blender:
            args.append("--profile")
        if profile_disable_denoise:
            args.append("--profile_disable_denoise")
        if profile_no_write:
            args.append("--profile_no_write")
        if skip_normalize:
            args.append("--skip_normalize")

        return_code = _run_blender_process(args, verbose_blender, blender_log, timeout * len(jobs))
        if return_code != 0 or not os.path.exists(records_path):
            error = (
                f"batch blender timed out after {timeout * len(jobs)} seconds"
                if return_code == -9 else
                f"batch blender failed with exit code {return_code}"
            )
            return [
                {
                    "sha256": job["sha256"],
                    "rendered": False,
                    "error": error,
                }
                for job in jobs
            ]

        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    by_sha = {record["sha256"]: record for record in records}
    checked_records = []
    for job in jobs:
        sha256 = job["sha256"]
        record = by_sha.get(sha256, {"sha256": sha256, "rendered": False, "error": "missing batch record"})
        if record.get("rendered") and not _render_complete(job["output_folder"], num_views):
            record = {
                "sha256": sha256,
                "rendered": False,
                "wall_seconds": record.get("wall_seconds"),
                "error": "render output incomplete after batch",
            }
        checked_records.append(record)
    return checked_records


def _foreach_instance(metadata: pd.DataFrame, dataset_root: str, func, max_workers: int, desc: str) -> pd.DataFrame:
    records = []
    items = metadata.to_dict("records")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            sha256 = item["sha256"]
            file_path = os.path.join(dataset_root, item["local_path"])
            futures[executor.submit(func, file_path, sha256)] = sha256
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sha256 = futures[future]
            try:
                record = future.result()
            except Exception as e:
                record = {"sha256": sha256, "rendered": False, "error": str(e)}
            if record is not None:
                records.append(record)

    return pd.DataFrame.from_records(records)


def _foreach_batch(
    metadata: pd.DataFrame,
    dataset_root: str,
    batch_size: int,
    output_dir: str,
    num_views: int,
    func,
    max_workers: int,
    desc: str,
) -> pd.DataFrame:
    records = []
    rows = metadata.to_dict("records")
    chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chunk in chunks:
            jobs = []
            views = _build_views(num_views)
            for item in chunk:
                sha256 = item["sha256"]
                jobs.append({
                    "sha256": sha256,
                    "object": os.path.join(dataset_root, item["local_path"]),
                    "output_folder": os.path.join(output_dir, "renders", sha256),
                    "views": views,
                })
            futures[executor.submit(func, jobs)] = [job["sha256"] for job in jobs]

        total = len(rows)
        with tqdm(total=total, desc=desc) as pbar:
            for future in as_completed(futures):
                try:
                    batch_records = future.result()
                except Exception as e:
                    batch_records = [
                        {"sha256": sha256, "rendered": False, "error": str(e)}
                        for sha256 in futures[future]
                    ]
                records.extend(batch_records)
                pbar.update(len(futures[future]))

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize original FaceScape meshes and render TRELLIS multi-view images"
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="FaceScape dataset root used with metadata.csv local_path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing metadata.csv and output renders/{sha256}")
    parser.add_argument("--instances", type=str, default=None, help="Comma-separated sha256 values or a file containing one sha256 per line")
    parser.add_argument("--num_views", type=int, default=150, help="Number of views to render")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--blender_batch_size", type=int, default=1,
                        help="Number of samples processed sequentially by one Blender process")
    parser.add_argument("--verbose_blender", action="store_true", help="Print Blender output, including Cycles GPU device selection logs")
    parser.add_argument("--profile_blender", action="store_true", help="Print filtered Blender timing profile logs")
    parser.add_argument("--blender_log", action="store_true", help="Print full raw Blender logs; very verbose")
    parser.add_argument("--profile_disable_denoise", action="store_true", help="Profiling only: disable Cycles denoising")
    parser.add_argument("--profile_no_write", action="store_true", help="Profiling only: time image saving outside write_still")
    parser.add_argument("--skip_normalize", action="store_true", help="Render meshes that are already normalized without applying another normalization")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-sample Blender timeout in seconds. Default is 300s based on the optimized no-denoise render timing.")
    opt = parser.parse_args()
    if opt.profile_blender:
        opt.verbose_blender = True
    if opt.blender_log:
        opt.verbose_blender = True

    os.makedirs(os.path.join(opt.output_dir, "renders"), exist_ok=True)
    dataset_root = os.path.abspath(os.path.expanduser(opt.dataset_root))
    if not os.path.exists(dataset_root):
        raise ValueError(f"dataset_root not found: {dataset_root}")

    print("Checking blender...", flush=True)
    _install_blender()

    metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise ValueError("metadata.csv not found")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError("metadata.csv must contain a sha256 column")
    if "local_path" not in metadata.columns:
        raise ValueError("metadata.csv must contain a local_path column")
    metadata = metadata[metadata["sha256"].notna()].copy()
    metadata = metadata[metadata["local_path"].notna()].copy()
    metadata["sha256"] = metadata["sha256"].astype(str)

    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r", encoding="utf-8") as f:
                instances = [line.strip() for line in f if line.strip()]
        else:
            instances = [item.strip() for item in opt.instances.split(",") if item.strip()]
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    for sha256 in copy.copy(metadata["sha256"].values):
        output_folder = os.path.join(opt.output_dir, "renders", sha256)
        if _render_complete(output_folder, opt.num_views):
            records.append({"sha256": sha256, "rendered": True})
            metadata = metadata[metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects...")

    render_kwargs = {
        "output_dir": opt.output_dir,
        "num_views": opt.num_views,
        "verbose_blender": opt.verbose_blender,
        "profile_blender": opt.profile_blender,
        "blender_log": opt.blender_log,
        "profile_disable_denoise": opt.profile_disable_denoise,
        "profile_no_write": opt.profile_no_write,
        "skip_normalize": opt.skip_normalize,
        "timeout": opt.timeout,
    }
    if opt.blender_batch_size <= 1:
        render_func = partial(_render, **render_kwargs)
        rendered = _foreach_instance(metadata, dataset_root, render_func, max_workers=opt.max_workers, desc="Rendering FaceScape objects")
    else:
        batch_func = partial(_render_batch, **render_kwargs)
        rendered = _foreach_batch(
            metadata,
            dataset_root,
            opt.blender_batch_size,
            opt.output_dir,
            opt.num_views,
            batch_func,
            max_workers=opt.max_workers,
            desc="Rendering FaceScape objects",
        )
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)], ignore_index=True)
    rendered.to_csv(os.path.join(opt.output_dir, f"rendered_{opt.rank}.csv"), index=False)
