#!/usr/bin/env python3
"""Audit SLat GT reconstruction for FaceScape overfit data.

This script is newly added for the SLat GT reconstruction audit. It bypasses
the SLat flow model entirely: it loads an existing latents/*.npz target,
constructs a SparseTensor, decodes it with the official SLat mesh decoder, and
exports meshes into an experiment directory.
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
from trellis.modules import sparse as sp


DEFAULT_SHA = "f674d4b2a1a9631d290345575d2dfd55419996523bc015e9c6c6c3b92a2984b9"
DEFAULT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a SLat GT latent and export its reconstruction mesh for audit."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/root/autodl-tmp/TRELLIS-new/datasets/Facescape/overfit_1"),
        help="Dataset root containing latents/<latent_model>/<sha>.npz.",
    )
    parser.add_argument("--sha256", default=DEFAULT_SHA, help="Sample sha256 to audit.")
    parser.add_argument(
        "--latent_model",
        default=DEFAULT_LATENT_MODEL,
        help="SLat latent model subdirectory under latents.",
    )
    parser.add_argument(
        "--decoder_path",
        default="/root/autodl-tmp/TRELLIS-new/microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
        help="from_pretrained-compatible SLat mesh decoder prefix.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/root/autodl-tmp/Stable3DGen/experiments/slat_gt_reconstruction_audit/overfit_1"),
        help="Directory where audit artifacts are written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for decoder inference.",
    )
    return parser.parse_args()


def load_slat(latent_path: Path, device: torch.device) -> sp.SparseTensor:
    data = np.load(latent_path)
    for key in ("coords", "feats"):
        if key not in data.files:
            raise KeyError(f"{latent_path} does not contain key '{key}'; keys={data.files}")

    coords = torch.from_numpy(data["coords"].astype(np.int32))
    feats = torch.from_numpy(data["feats"].astype(np.float32))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3), got {tuple(coords.shape)}")
    if feats.ndim != 2:
        raise ValueError(f"Expected feats shape (N,C), got {tuple(feats.shape)}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"coords/feats length mismatch: {coords.shape[0]} vs {feats.shape[0]}")

    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    batched_coords = torch.cat([batch, coords], dim=1).to(device)
    feats = feats.to(device)
    slat = sp.SparseTensor(coords=batched_coords, feats=feats)
    slat._shape = torch.Size([1, feats.shape[1]])
    slat.register_spatial_cache("layout", [slice(0, feats.shape[0])])
    return slat


def mesh_result_to_trimesh(mesh_result):
    import trimesh

    vertices = mesh_result.vertices.detach().float().cpu().numpy()
    faces = mesh_result.faces.detach().long().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def rotate_mesh_x_positive_90(mesh):
    """Match the Stable3DGen CLI export pose for easier visual comparison."""
    import trimesh

    rotation = trimesh.transformations.rotation_matrix(
        angle=np.deg2rad(90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    mesh.apply_transform(rotation)
    return mesh


@torch.no_grad()
def main() -> None:
    args = parse_args()
    latent_path = args.dataset_root / "latents" / args.latent_model / f"{args.sha256}.npz"
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    slat = load_slat(latent_path, device)

    # New audit path: load only the SLat mesh decoder so flow training/inference cannot affect the result.
    decoder = models.from_pretrained(args.decoder_path).eval().to(device)
    meshes = decoder(slat)
    if not meshes:
        raise RuntimeError("SLat mesh decoder returned no meshes.")
    mesh_result = meshes[0]
    if not getattr(mesh_result, "success", False):
        raise RuntimeError("SLat mesh decoder returned an empty mesh.")

    sample_dir = args.output_dir / args.sha256
    sample_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = sample_dir / "slat_gt_recon_mesh.ply"
    cli_pose_mesh_path = sample_dir / "slat_gt_recon_mesh_cli_pose.ply"
    stats_path = sample_dir / "slat_gt_recon_stats.json"

    mesh = mesh_result_to_trimesh(mesh_result)
    mesh.export(mesh_path)
    cli_pose_mesh = mesh.copy()
    rotate_mesh_x_positive_90(cli_pose_mesh)
    cli_pose_mesh.export(cli_pose_mesh_path)

    stats = {
        "sha256": args.sha256,
        "latent_path": str(latent_path),
        "decoder_path": args.decoder_path,
        "num_sparse_points": int(slat.feats.shape[0]),
        "feature_channels": int(slat.feats.shape[1]),
        "feats_mean": float(slat.feats.detach().float().mean().cpu()),
        "feats_std": float(slat.feats.detach().float().std().cpu()),
        "coords_min": [int(v) for v in slat.coords[:, 1:].detach().cpu().min(dim=0).values.tolist()],
        "coords_max": [int(v) for v in slat.coords[:, 1:].detach().cpu().max(dim=0).values.tolist()],
        "mesh_success": bool(mesh_result.success),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "mesh_path": str(mesh_path),
        "cli_pose_mesh_path": str(cli_pose_mesh_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote mesh: {mesh_path}")
    print(f"[OK] wrote CLI-pose mesh: {cli_pose_mesh_path}")
    print(f"[OK] wrote stats: {stats_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
