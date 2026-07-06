#!/usr/bin/env python3
"""Export random FaceScape train GT reconstructions from SS and SLat latents.

This diagnostic script is newly added for latent-capacity auditing. It bypasses
all flow models: for each sampled training SHA it decodes the stored ss_latent
and slat_latent directly, then writes the GT reconstruction meshes into one
standalone experiment directory so training targets can be inspected in bulk.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# New bulk audit convenience: keep the script runnable from fine_tuning/ or repo root.
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


DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/TRELLIS-new/datasets/Facescape/train")
DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/Stable3DGen/experiments/train_random100_gt_reconstruction"
)
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
        description="Randomly sample training latents and export SS/SLat GT reconstructions."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--caption_contains",
        default=None,
        help="Optional substring filter applied to metadata.csv captions before random sampling.",
    )
    parser.add_argument("--ss_latent_model", default=DEFAULT_SS_LATENT_MODEL)
    parser.add_argument("--slat_latent_model", default=DEFAULT_SLAT_LATENT_MODEL)
    parser.add_argument("--ss_decoder_path", default=DEFAULT_SS_DECODER)
    parser.add_argument("--slat_decoder_path", default=DEFAULT_SLAT_DECODER)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory manifest and sample files.",
    )
    parser.add_argument(
        "--make_tar",
        action="store_true",
        help="Create <output_dir>.tar after all samples are exported.",
    )
    return parser.parse_args()


def list_latent_shas(latent_dir: Path) -> set[str]:
    if not latent_dir.is_dir():
        raise FileNotFoundError(latent_dir)
    return {path.stem for path in latent_dir.glob("*.npz")}


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return pd.read_csv(metadata_path)


def select_samples(args: argparse.Namespace) -> pd.DataFrame:
    metadata = load_metadata(args.dataset_root)
    ss_dir = args.dataset_root / "ss_latents" / args.ss_latent_model
    slat_dir = args.dataset_root / "latents" / args.slat_latent_model
    common_shas = list_latent_shas(ss_dir) & list_latent_shas(slat_dir)

    metadata = metadata[metadata["sha256"].isin(common_shas)].copy()
    if args.caption_contains is not None:
        if "captions" not in metadata.columns:
            raise KeyError("metadata.csv has no captions column for --caption_contains")
        # New audit filter: restrict the random draw to a caption subset, e.g. 1_neutral.
        metadata = metadata[
            metadata["captions"].astype(str).str.contains(args.caption_contains, na=False)
        ]
    if "ss_latent_ss_enc_conv3d_16l8_fp16" in metadata.columns:
        metadata = metadata[metadata["ss_latent_ss_enc_conv3d_16l8_fp16"] == True]
    if "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16" in metadata.columns:
        metadata = metadata[metadata["latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"] == True]

    if len(metadata) < args.num_samples:
        raise ValueError(
            f"Only {len(metadata)} samples have both latents; requested {args.num_samples}."
        )

    rng = random.Random(args.seed)
    selected_shas = rng.sample(sorted(metadata["sha256"].tolist()), args.num_samples)
    selected = metadata.set_index("sha256").loc[selected_shas].reset_index()
    return selected


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Pass --overwrite to refresh this audit output."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)


@torch.no_grad()
def export_one_sample(
    sha: str,
    row: pd.Series,
    args: argparse.Namespace,
    ss_decoder: torch.nn.Module,
    slat_decoder: torch.nn.Module,
    device: torch.device,
) -> dict:
    sample_dir = args.output_dir / "samples" / sha
    sample_dir.mkdir(parents=True, exist_ok=True)

    ss_latent_path = args.dataset_root / "ss_latents" / args.ss_latent_model / f"{sha}.npz"
    slat_latent_path = args.dataset_root / "latents" / args.slat_latent_model / f"{sha}.npz"

    # New bulk audit path: decode stored SS target directly, so flow inference quality is irrelevant.
    ss_latent = load_latent(ss_latent_path).to(device)
    ss_logits = ss_decoder(ss_latent)
    occupancy = (ss_logits[0, 0].detach().float().cpu().numpy() > args.threshold)
    coords = np.argwhere(occupancy)
    ss_mesh = occupancy_to_mesh(occupancy)
    ss_mesh_path = sample_dir / "ss_gt_recon_mesh.ply"
    ss_points_path = sample_dir / "ss_gt_recon_occupied_points.ply"
    ss_mesh.export(ss_mesh_path)
    write_occupancy_points(ss_points_path, coords, occupancy.shape[0])

    # New bulk audit path: decode stored SLat target directly with the official mesh decoder.
    slat = load_slat(slat_latent_path, device)
    mesh_results = slat_decoder(slat)
    if not mesh_results or not getattr(mesh_results[0], "success", False):
        raise RuntimeError(f"SLat decoder returned no mesh for {sha}")
    slat_mesh = mesh_result_to_trimesh(mesh_results[0])
    slat_mesh_path = sample_dir / "slat_gt_recon_mesh.ply"
    slat_cli_pose_path = sample_dir / "slat_gt_recon_mesh_cli_pose.ply"
    slat_mesh.export(slat_mesh_path)
    slat_cli_pose_mesh = slat_mesh.copy()
    rotate_mesh_x_positive_90(slat_cli_pose_mesh)
    slat_cli_pose_mesh.export(slat_cli_pose_path)

    stats = {
        "sha256": sha,
        "caption": row.get("captions", ""),
        "metadata_num_voxels": int(row["num_voxels"]) if "num_voxels" in row else None,
        "ss_latent_path": str(ss_latent_path),
        "slat_latent_path": str(slat_latent_path),
        "ss_occupied_voxels": int(coords.shape[0]),
        "ss_latent_shape": list(ss_latent.shape),
        "ss_logit_min": float(ss_logits.detach().float().min().cpu()),
        "ss_logit_max": float(ss_logits.detach().float().max().cpu()),
        "ss_mesh_path": str(ss_mesh_path),
        "ss_points_path": str(ss_points_path),
        "slat_sparse_points": int(slat.feats.shape[0]),
        "slat_feature_channels": int(slat.feats.shape[1]),
        "slat_coords_min": [
            int(v) for v in slat.coords[:, 1:].detach().cpu().min(dim=0).values.tolist()
        ],
        "slat_coords_max": [
            int(v) for v in slat.coords[:, 1:].detach().cpu().max(dim=0).values.tolist()
        ],
        "slat_mesh_vertices": int(slat_mesh.vertices.shape[0]),
        "slat_mesh_faces": int(slat_mesh.faces.shape[0]),
        "slat_mesh_path": str(slat_mesh_path),
        "slat_cli_pose_mesh_path": str(slat_cli_pose_path),
    }
    (sample_dir / "gt_recon_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Keep the selected row next to the meshes so each folder is self-describing.
    row_payload = {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
    (sample_dir / "metadata_row.json").write_text(
        json.dumps(row_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def write_selection_files(output_dir: Path, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    selected.to_csv(output_dir / "selected_samples.csv", index=False)
    (output_dir / "selected_sha256.txt").write_text(
        "\n".join(selected["sha256"].tolist()) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(output_dir),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "caption_contains": args.caption_contains,
        "ss_latent_model": args.ss_latent_model,
        "slat_latent_model": args.slat_latent_model,
        "ss_decoder_path": args.ss_decoder_path,
        "slat_decoder_path": args.slat_decoder_path,
        "threshold": float(args.threshold),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_summary(output_dir: Path, stats_rows: list[dict]) -> None:
    summary_json = output_dir / "summary_stats.json"
    summary_json.write_text(json.dumps(stats_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_csv = output_dir / "summary_stats.csv"
    if not stats_rows:
        return
    keys = sorted({key for row in stats_rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats_rows)


def make_tar(output_dir: Path) -> Path:
    tar_path = output_dir.with_suffix(".tar")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    return tar_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    prepare_output_dir(args.output_dir, args.overwrite)
    selected = select_samples(args)
    write_selection_files(args.output_dir, selected, args)

    # Load decoders once for the full random-100 audit; this keeps the job practical.
    ss_decoder = models.from_pretrained(args.ss_decoder_path).eval().to(device)
    slat_decoder = models.from_pretrained(args.slat_decoder_path).eval().to(device)

    stats_rows: list[dict] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Exporting GT recon"):
        sha = row["sha256"]
        stats_rows.append(export_one_sample(sha, row, args, ss_decoder, slat_decoder, device))
        torch.cuda.empty_cache()

    write_summary(args.output_dir, stats_rows)
    if args.make_tar:
        tar_path = make_tar(args.output_dir)
        print(f"[OK] wrote tar: {tar_path}")
    print(f"[OK] wrote output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
