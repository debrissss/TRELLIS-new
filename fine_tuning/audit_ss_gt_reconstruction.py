#!/usr/bin/env python3
"""Audit sparse-structure GT latent reconstruction for FaceScape overfit data.

This script is newly added for the GT reconstruction audit. It bypasses the
SS flow model entirely: it loads an existing ss_latents/*.npz target, decodes it
with the official SS decoder, thresholds the occupancy logits exactly like the
TRELLIS/Stable3DGen pipelines, and exports a mesh into an experiment directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# New audit script convenience: make it runnable directly from fine_tuning/
# without requiring callers to remember PYTHONPATH=/root/autodl-tmp/TRELLIS-new.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import models


DEFAULT_SHA = "f674d4b2a1a9631d290345575d2dfd55419996523bc015e9c6c6c3b92a2984b9"
DEFAULT_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode an SS GT latent and export its reconstruction mesh for audit."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/root/autodl-tmp/TRELLIS-new/datasets/Facescape/overfit_1"),
        help="Dataset root containing ss_latents/<latent_model>/<sha>.npz.",
    )
    parser.add_argument("--sha256", default=DEFAULT_SHA, help="Sample sha256 to audit.")
    parser.add_argument(
        "--latent_model",
        default=DEFAULT_LATENT_MODEL,
        help="SS latent model subdirectory under ss_latents.",
    )
    parser.add_argument(
        "--decoder_path",
        default="/root/autodl-tmp/TRELLIS-new/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        help="from_pretrained-compatible SS decoder prefix.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/root/autodl-tmp/Stable3DGen/experiments/ss_gt_reconstruction_audit/overfit_1"),
        help="Directory where audit artifacts are written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Occupancy logit threshold. TRELLIS pipelines use decoder(z_s) > 0.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for decoder inference.",
    )
    return parser.parse_args()


def load_latent(latent_path: Path) -> torch.Tensor:
    data = np.load(latent_path)
    if "mean" not in data.files:
        raise KeyError(f"{latent_path} does not contain key 'mean'; keys={data.files}")
    latent = torch.from_numpy(data["mean"]).float().unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(f"Expected latent shape (1,C,D,H,W), got {tuple(latent.shape)}")
    return latent


def occupancy_to_mesh(occupancy: np.ndarray):
    """Convert a 64^3 binary occupancy grid to a simple marching-cubes mesh."""
    from skimage import measure
    import trimesh

    if occupancy.ndim != 3:
        raise ValueError(f"Expected 3D occupancy, got {occupancy.shape}")
    if occupancy.sum() == 0:
        raise ValueError("Decoded occupancy is empty; cannot export mesh.")

    # Padding keeps marching cubes stable when occupied voxels touch the border.
    padded = np.pad(occupancy.astype(np.float32), 1, mode="constant", constant_values=0.0)
    vertices, faces, _normals, _values = measure.marching_cubes(padded, level=0.5)
    vertices -= 1.0
    resolution = np.array(occupancy.shape, dtype=np.float32)
    vertices = (vertices + 0.5) / resolution - 0.5
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def write_occupancy_points(path: Path, coords: np.ndarray, resolution: int) -> None:
    """Write occupied voxel centers as an auxiliary point cloud for debugging."""
    points = (coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    latent_path = args.dataset_root / "ss_latents" / args.latent_model / f"{args.sha256}.npz"
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    latent = load_latent(latent_path).to(device)

    # New audit path: load only the SS decoder so flow training/inference cannot affect the result.
    decoder = models.from_pretrained(args.decoder_path).eval().to(device)
    logits = decoder(latent)
    occupancy = (logits[0, 0].detach().float().cpu().numpy() > args.threshold)
    coords = np.argwhere(occupancy)

    sample_dir = args.output_dir / args.sha256
    sample_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = sample_dir / "ss_gt_recon_mesh.ply"
    points_path = sample_dir / "ss_gt_recon_occupied_points.ply"
    stats_path = sample_dir / "ss_gt_recon_stats.json"

    mesh = occupancy_to_mesh(occupancy)
    mesh.export(mesh_path)
    write_occupancy_points(points_path, coords, occupancy.shape[0])

    stats = {
        "sha256": args.sha256,
        "latent_path": str(latent_path),
        "decoder_path": args.decoder_path,
        "threshold": args.threshold,
        "latent_shape": list(latent.shape),
        "latent_mean": float(latent.detach().float().mean().cpu()),
        "latent_std": float(latent.detach().float().std().cpu()),
        "logit_min": float(logits.detach().float().min().cpu()),
        "logit_max": float(logits.detach().float().max().cpu()),
        "logit_mean": float(logits.detach().float().mean().cpu()),
        "occupied_voxels": int(coords.shape[0]),
        "occupancy_shape": list(occupancy.shape),
        "mesh_path": str(mesh_path),
        "points_path": str(points_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote mesh: {mesh_path}")
    print(f"[OK] wrote occupied points: {points_path}")
    print(f"[OK] wrote stats: {stats_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
