#!/usr/bin/env python3
"""Process user-truncated head meshes and export TRELLIS GT reconstructions.

This diagnostic script is newly added for the manually truncated 1_neutral
meshes. It uses fine_tuning/facescape_render.py for the normalization/render
stage, then reuses the existing dataset_toolkits preprocessing stages to
produce SS/SLat latents and decoder-only GT meshes. Final artifacts are copied
back beside each source mesh with a mesh_truncated_ prefix so the original
files and earlier GT exports remain protected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# New bulk diagnostic convenience: make this runnable from repo root or fine_tuning/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.audit_slat_gt_reconstruction import (  # noqa: E402
    load_slat,
    mesh_result_to_trimesh,
    rotate_mesh_x_positive_90,
)
from fine_tuning.audit_ss_gt_reconstruction import (  # noqa: E402
    load_latent,
    occupancy_to_mesh,
    write_occupancy_points,
)
from trellis import models  # noqa: E402


DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/TRELLIS-new/train_1_neutral_turncate_gt_reconstruction"
)
DEFAULT_PYTHON = Path("/root/autodl-tmp/mamba_envs/trellis5090/bin/python")
DEFAULT_PROCESSING_NAME = "_trellis_truncated_processing"
DEFAULT_SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
DEFAULT_SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
DEFAULT_SS_DECODER = (
    "/root/autodl-tmp/TRELLIS-new/microsoft/TRELLIS-image-large/ckpts/"
    "ss_dec_conv3d_16l8_fp16"
)
DEFAULT_SLAT_DECODER = (
    "/root/autodl-tmp/TRELLIS-new/microsoft/TRELLIS-image-large/ckpts/"
    "slat_dec_mesh_swin8_B_64l8m256c_fp16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process per-SHA mesh_truncated.ply files and export GT recon meshes."
    )
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--processing_dir", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--num_views", type=int, default=150)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--blender_batch_size", type=int, default=4)
    parser.add_argument("--render_timeout", type=float, default=300.0)
    parser.add_argument(
        "--render_with_denoise",
        action="store_true",
        help="Keep Blender/Cycles denoising enabled. Default is disabled for faster truncated-mesh preprocessing.",
    )
    parser.add_argument("--voxel_timeout", type=float, default=60.0)
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--ss_latent_model", default=DEFAULT_SS_LATENT_MODEL)
    parser.add_argument("--slat_latent_model", default=DEFAULT_SLAT_LATENT_MODEL)
    parser.add_argument("--ss_decoder_path", default=DEFAULT_SS_DECODER)
    parser.add_argument("--slat_decoder_path", default=DEFAULT_SLAT_DECODER)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate preprocessing outputs and overwrite mesh_truncated_* artifacts.",
    )
    parser.add_argument(
        "--keep_existing_render",
        action="store_true",
        help="Let facescape_render.py reuse complete existing renders even with --overwrite.",
    )
    return parser.parse_args()


def env_for_subprocess() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "ATTN_BACKEND": env.get("ATTN_BACKEND", "sdpa"),
            "SPARSE_ATTN_BACKEND": env.get("SPARSE_ATTN_BACKEND", "flash_attn"),
            "SPCONV_ALGO": env.get("SPCONV_ALGO", "native"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "8"),
        }
    )
    return env


def discover_samples(source_root: Path) -> list[dict[str, str]]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    samples = []
    for sample_dir in sorted(source_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        if sample_dir.name.startswith(".") or sample_dir.name.startswith("_"):
            continue
        mesh_path = sample_dir / "mesh_truncated.ply"
        if mesh_path.is_file():
            samples.append(
                {
                    "sha256": sample_dir.name,
                    "local_path": f"{sample_dir.name}/mesh_truncated.ply",
                    "captions": f"{sample_dir.name}_mesh_truncated",
                    "source_dir": str(sample_dir),
                    "source_mesh": str(mesh_path),
                }
            )
    if not samples:
        raise RuntimeError(f"No <sha>/mesh_truncated.ply files found under {source_root}")
    return samples


def write_initial_metadata(processing_dir: Path, samples: list[dict[str, str]]) -> None:
    processing_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        rows.append(
            {
                "sha256": sample["sha256"],
                "local_path": sample["local_path"],
                "captions": sample["captions"],
                "rendered": False,
                "voxelized": False,
                "num_voxels": 0,
                "cond_rendered": False,
                "feature_dinov2_vitl14_reg": False,
                f"latent_{DEFAULT_SLAT_LATENT_MODEL}": False,
                f"ss_latent_{DEFAULT_SS_LATENT_MODEL}": False,
                "aesthetic_score": 5.0,
                "source_dir": sample["source_dir"],
                "source_mesh": sample["source_mesh"],
            }
        )
    pd.DataFrame(rows).to_csv(processing_dir / "metadata.csv", index=False)
    (processing_dir / "instances.txt").write_text(
        "\n".join(sample["sha256"] for sample in samples) + "\n", encoding="utf-8"
    )


def read_stage_csv(processing_dir: Path, prefix: str) -> pd.DataFrame:
    paths = sorted(processing_dir.glob(f"{prefix}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.csv found in {processing_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def update_metadata_from_stage(processing_dir: Path, stage: str) -> None:
    metadata_path = processing_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    stage_df = read_stage_csv(processing_dir, stage)
    if "sha256" not in stage_df.columns:
        raise ValueError(f"{stage}_*.csv has no sha256 column")
    stage_df = stage_df.drop_duplicates("sha256", keep="last").set_index("sha256")
    metadata = metadata.set_index("sha256")
    for column in stage_df.columns:
        if column not in metadata.columns:
            metadata[column] = None
        metadata.loc[stage_df.index, column] = stage_df[column]
    metadata.reset_index().to_csv(metadata_path, index=False)


def run_command(args: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(args), flush=True)
    subprocess.run(args, cwd=str(cwd), env=env_for_subprocess(), check=True)


def remove_stage_outputs(processing_dir: Path, samples: list[dict[str, str]], keep_existing_render: bool) -> None:
    names = [
        "voxels",
        "features",
        "latents",
        "ss_latents",
    ]
    if not keep_existing_render:
        names.insert(0, "renders")
    for name in names:
        path = processing_dir / name
        if path.exists():
            shutil.rmtree(path)
    for pattern in [
        "rendered_*.csv",
        "voxelized_*.csv",
        "feature_*.csv",
        "latent_*.csv",
        "ss_latent_*.csv",
    ]:
        for path in processing_dir.glob(pattern):
            path.unlink()
    write_initial_metadata(processing_dir, samples)


def run_preprocessing(args: argparse.Namespace, processing_dir: Path) -> None:
    instances_path = processing_dir / "instances.txt"
    render_cmd = [
        str(args.python),
        "fine_tuning/facescape_render.py",
        "--dataset_root",
        str(args.source_root),
        "--output_dir",
        str(processing_dir),
        "--instances",
        str(instances_path),
        "--num_views",
        str(args.num_views),
        "--max_workers",
        str(args.max_workers),
        "--blender_batch_size",
        str(args.blender_batch_size),
        "--timeout",
        str(args.render_timeout),
    ]
    if not args.render_with_denoise:
        # New truncated-mesh audit default: disable Cycles denoising during facescape_render
        # to speed up normalization/rendering; geometry preprocessing does not rely on denoised RGB.
        render_cmd.append("--profile_disable_denoise")
    run_command(render_cmd, REPO_ROOT)
    update_metadata_from_stage(processing_dir, "rendered")

    run_command(
        [
            str(args.python),
            "fine_tuning/voxelize.py",
            "SingleMesh",
            "--output_dir",
            str(processing_dir),
            "--instances",
            str(instances_path),
            "--max_workers",
            str(args.max_workers),
            "--timeout",
            str(args.voxel_timeout),
        ],
        REPO_ROOT,
    )
    update_metadata_from_stage(processing_dir, "voxelized")

    run_command(
        [
            str(args.python),
            "fine_tuning/facescape_extract_feature.py",
            "--output_dir",
            str(processing_dir),
            "--instances",
            str(instances_path),
            "--batch_size",
            str(args.feature_batch_size),
            "--overwrite",
        ],
        REPO_ROOT,
    )
    update_metadata_from_stage(processing_dir, "feature_dinov2_vitl14_reg")

    run_command(
        [
            str(args.python),
            "dataset_toolkits/encode_latent.py",
            "--output_dir",
            str(processing_dir),
            "--enc_pretrained",
            str(REPO_ROOT / "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"),
            "--instances",
            str(instances_path),
        ],
        REPO_ROOT,
    )
    update_metadata_from_stage(processing_dir, f"latent_{args.slat_latent_model}")

    run_command(
        [
            str(args.python),
            "dataset_toolkits/encode_ss_latent.py",
            "--output_dir",
            str(processing_dir),
            "--enc_pretrained",
            str(REPO_ROOT / "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"),
            "--instances",
            str(instances_path),
        ],
        REPO_ROOT,
    )
    update_metadata_from_stage(processing_dir, f"ss_latent_{args.ss_latent_model}")


def copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


@torch.no_grad()
def decode_and_copy_outputs(args: argparse.Namespace, processing_dir: Path, samples: list[dict[str, str]]) -> None:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ss_decoder = models.from_pretrained(args.ss_decoder_path).eval().to(device)
    slat_decoder = models.from_pretrained(args.slat_decoder_path).eval().to(device)

    stats_rows = []
    for sample in tqdm(samples, desc="Decoding truncated GT"):
        sha = sample["sha256"]
        source_dir = Path(sample["source_dir"])
        ss_latent_path = processing_dir / "ss_latents" / args.ss_latent_model / f"{sha}.npz"
        slat_latent_path = processing_dir / "latents" / args.slat_latent_model / f"{sha}.npz"

        copy_required(ss_latent_path, source_dir / "mesh_truncated_ss_latent.npz")
        copy_required(slat_latent_path, source_dir / "mesh_truncated_slat_latent.npz")
        copy_required(
            processing_dir / "renders" / sha / "mesh.ply",
            source_dir / "mesh_truncated_normalized_mesh.ply",
        )

        # New truncated-mesh audit path: decode stored SS target directly.
        ss_latent = load_latent(ss_latent_path).to(device)
        ss_logits = ss_decoder(ss_latent)
        occupancy = (ss_logits[0, 0].detach().float().cpu().numpy() > args.threshold)
        coords = np.argwhere(occupancy)
        ss_mesh = occupancy_to_mesh(occupancy)
        ss_mesh_path = source_dir / "mesh_truncated_ss_gt_recon_mesh.ply"
        ss_points_path = source_dir / "mesh_truncated_ss_gt_recon_occupied_points.ply"
        ss_mesh.export(ss_mesh_path)
        write_occupancy_points(ss_points_path, coords, occupancy.shape[0])

        # New truncated-mesh audit path: decode stored SLat target directly.
        slat = load_slat(slat_latent_path, device)
        mesh_results = slat_decoder(slat)
        if not mesh_results or not getattr(mesh_results[0], "success", False):
            raise RuntimeError(f"SLat decoder returned no mesh for {sha}")
        slat_mesh = mesh_result_to_trimesh(mesh_results[0])
        slat_mesh_path = source_dir / "mesh_truncated_slat_gt_recon_mesh.ply"
        slat_cli_pose_path = source_dir / "mesh_truncated_slat_gt_recon_mesh_cli_pose.ply"
        slat_mesh.export(slat_mesh_path)
        slat_cli_pose_mesh = slat_mesh.copy()
        rotate_mesh_x_positive_90(slat_cli_pose_mesh)
        slat_cli_pose_mesh.export(slat_cli_pose_path)

        voxel_path = processing_dir / "voxels" / f"{sha}.ply"
        num_voxels = None
        if voxel_path.is_file():
            import utils3d

            num_voxels = int(len(utils3d.io.read_ply(str(voxel_path))[0]))

        stats = {
            "sha256": sha,
            "source_mesh": sample["source_mesh"],
            "processing_dir": str(processing_dir),
            "normalized_mesh": str(source_dir / "mesh_truncated_normalized_mesh.ply"),
            "ss_latent": str(source_dir / "mesh_truncated_ss_latent.npz"),
            "slat_latent": str(source_dir / "mesh_truncated_slat_latent.npz"),
            "num_voxels": num_voxels,
            "ss_occupied_voxels": int(coords.shape[0]),
            "ss_mesh": str(ss_mesh_path),
            "ss_points": str(ss_points_path),
            "slat_sparse_points": int(slat.feats.shape[0]),
            "slat_mesh_vertices": int(slat_mesh.vertices.shape[0]),
            "slat_mesh_faces": int(slat_mesh.faces.shape[0]),
            "slat_mesh": str(slat_mesh_path),
            "slat_cli_pose_mesh": str(slat_cli_pose_path),
        }
        (source_dir / "mesh_truncated_gt_recon_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "purpose": "new truncated mesh TRELLIS preprocessing audit",
            "source_mesh": sample["source_mesh"],
            "num_views": args.num_views,
            "render_with_denoise": args.render_with_denoise,
            "ss_decoder_path": args.ss_decoder_path,
            "slat_decoder_path": args.slat_decoder_path,
            "threshold": args.threshold,
        }
        (source_dir / "mesh_truncated_processing_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        stats_rows.append(stats)
        torch.cuda.empty_cache()

    write_summary(processing_dir, stats_rows)


def write_summary(processing_dir: Path, stats_rows: list[dict]) -> None:
    summary_json = processing_dir / "summary.json"
    summary_csv = processing_dir / "summary.csv"
    summary_json.write_text(json.dumps(stats_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not stats_rows:
        return
    keys = sorted({key for row in stats_rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats_rows)


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    processing_dir = (
        args.processing_dir.resolve()
        if args.processing_dir is not None
        else args.source_root / DEFAULT_PROCESSING_NAME
    )
    samples = discover_samples(args.source_root)

    if args.overwrite:
        remove_stage_outputs(processing_dir, samples, args.keep_existing_render)
    else:
        write_initial_metadata(processing_dir, samples)

    run_preprocessing(args, processing_dir)
    decode_and_copy_outputs(args, processing_dir, samples)
    print(f"[OK] processed {len(samples)} truncated meshes")
    print(f"[OK] summary: {processing_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
