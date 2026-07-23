"""Evaluate SS flow sparse-structure sampling on fixed image conditions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import utils3d
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate_ss_enc_dec_reconstruction import compute_binary_metrics
from trellis import models
from trellis.pipelines import TrellisImageTo3DPipeline


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUMMARY_METRICS = (
    "iou",
    "dice_f1",
    "occupancy_ratio",
    "gt_occupied_voxels",
    "predicted_occupied_voxels",
)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    return state


def load_pipeline(args: argparse.Namespace, device: torch.device) -> TrellisImageTo3DPipeline:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline_path)
    pipeline.to(device)

    flow_state = load_state_dict(Path(args.flow_ckpt))
    pipeline.models["sparse_structure_flow_model"].load_state_dict(flow_state)
    pipeline.models["sparse_structure_flow_model"].eval()

    decoder_state = load_state_dict(Path(args.decoder_ckpt))
    pipeline.models["sparse_structure_decoder"].load_state_dict(decoder_state)
    pipeline.models["sparse_structure_decoder"].eval()
    return pipeline


def load_encoder(args: argparse.Namespace, device: torch.device) -> torch.nn.Module | None:
    if args.encoder_ckpt is None and args.encoder_config is None:
        return None
    if args.encoder_ckpt is None or args.encoder_config is None:
        raise ValueError("--encoder_ckpt and --encoder_config must be provided together")
    with Path(args.encoder_config).open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    encoder_config = config.get("models", {}).get("encoder")
    if encoder_config is None:
        raise KeyError(f"Config is missing models.encoder: {args.encoder_config}")
    encoder = getattr(models, encoder_config["name"])(**encoder_config["args"]).to(device)
    encoder.load_state_dict(load_state_dict(Path(args.encoder_ckpt)))
    encoder.eval()
    return encoder


def truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def list_condition_images(root: Path, sha: str) -> list[Path]:
    cond_root = root / "renders_cond" / sha
    if not cond_root.is_dir():
        return []
    return sorted(
        path
        for path in cond_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_samples(data_root: Path, num_samples: int, seed: int, min_aesthetic_score: float | None) -> list[str]:
    metadata_path = data_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {"sha256", "voxelized", "cond_rendered"}
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"metadata.csv is missing columns: {missing_columns}")
    selected = metadata[metadata["voxelized"].apply(truthy) & metadata["cond_rendered"].apply(truthy)].copy()
    if min_aesthetic_score is not None:
        if "aesthetic_score" not in selected.columns:
            raise ValueError("metadata.csv is missing aesthetic_score for --min_aesthetic_score")
        selected = selected[selected["aesthetic_score"] >= min_aesthetic_score].copy()

    def has_required_files(sha: str) -> bool:
        return (data_root / "voxels" / f"{sha}.ply").is_file() and bool(list_condition_images(data_root, sha))

    selected = selected[selected["sha256"].apply(has_required_files)].copy()
    if len(selected) == 0:
        raise ValueError(f"No valid samples with voxels and condition images under {data_root}")
    if num_samples > 0 and len(selected) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but only {len(selected)} are valid")
    if num_samples > 0:
        selected = selected.sample(n=num_samples, random_state=seed)
    return selected["sha256"].tolist()


def load_gt_voxel(path: Path, resolution: int) -> torch.Tensor:
    positions = utils3d.io.read_ply(str(path))[0]
    coords = ((torch.tensor(positions).float() + 0.5) * resolution).long()
    coords = coords.clamp(0, resolution - 1)
    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.bool)
    if coords.numel() > 0:
        grid[:, coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


def coords_to_voxel(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.bool)
    if coords.numel() == 0:
        return grid
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected coords shape (N,4), got {tuple(coords.shape)}")
    sample_coords = coords[coords[:, 0] == 0][:, 1:].long().cpu()
    if sample_coords.numel() > 0:
        sample_coords = sample_coords.clamp(0, resolution - 1)
        grid[:, sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]] = True
    return grid


def write_occupied_points_ply(path: Path, occupancy: torch.Tensor) -> None:
    if occupancy.ndim == 4:
        occupancy = occupancy[0]
    coords = torch.nonzero(occupancy.bool(), as_tuple=False).cpu().numpy()
    resolution = occupancy.shape[-1]
    points = (coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(points)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("end_header\n")
        for x, y, z in points:
            fp.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], empty_threshold: int, overfull_ratio: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "empty_threshold": empty_threshold,
        "overfull_ratio": overfull_ratio,
        "empty_pred_count": int(sum(bool(row["is_empty_pred"]) for row in rows)),
        "overfull_pred_count": int(sum(bool(row["is_overfull_pred"]) for row in rows)),
    }
    summary["empty_pred_rate"] = summary["empty_pred_count"] / len(rows) if rows else float("nan")
    summary["overfull_pred_rate"] = summary["overfull_pred_count"] / len(rows) if rows else float("nan")

    for name in SUMMARY_METRICS:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[name] = {key: float("nan") for key in ("mean", "median", "std", "p10", "p90", "min", "max")}
        else:
            summary[name] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "std": float(np.std(finite)),
                "p10": float(np.percentile(finite, 10)),
                "p90": float(np.percentile(finite, 90)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
    return summary


def summarize_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for name in ("iou", "dice_f1", "occupancy_ratio_absolute_error"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    flat: dict[str, Any] = {
        "num_samples": summary["num_samples"],
        "empty_pred_count": summary["empty_pred_count"],
        "empty_pred_rate": summary["empty_pred_rate"],
        "overfull_pred_count": summary["overfull_pred_count"],
        "overfull_pred_rate": summary["overfull_pred_rate"],
    }
    for metric in SUMMARY_METRICS:
        for stat_name, value in summary[metric].items():
            flat[f"{metric}_{stat_name}"] = value
    write_rows_csv(path, [flat])


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples = select_samples(
        data_root=data_root,
        num_samples=args.num_samples,
        seed=args.seed,
        min_aesthetic_score=args.min_aesthetic_score,
    )
    pipeline = load_pipeline(args, device=device)
    encoder = load_encoder(args, device=device)
    resolution = args.resolution

    rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    sampler_params = {}
    if args.steps is not None:
        sampler_params["steps"] = args.steps
    if args.cfg_strength is not None:
        sampler_params["cfg_strength"] = args.cfg_strength
    if args.cfg_interval is not None:
        sampler_params["cfg_interval"] = tuple(args.cfg_interval)
    if args.rescale_t is not None:
        sampler_params["rescale_t"] = args.rescale_t

    for index, sha in enumerate(samples):
        sample_seed = args.seed + index
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        cond_image_path = list_condition_images(data_root, sha)[0]
        image = Image.open(cond_image_path)
        image = pipeline.preprocess_image(image) if args.preprocess_image else image.resize((518, 518), Image.LANCZOS)
        cond = pipeline.get_cond([image])
        coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params=sampler_params)
        pred = coords_to_voxel(coords, resolution=resolution)
        gt_path = data_root / "voxels" / f"{sha}.ply"
        gt = load_gt_voxel(gt_path, resolution=resolution)
        metrics = compute_binary_metrics(pred, gt)
        pred_count = int(metrics["predicted_occupied_voxels"])
        gt_count = int(metrics["gt_occupied_voxels"])
        is_empty = pred_count <= args.empty_threshold
        occupancy_ratio = metrics["occupancy_ratio"]
        is_overfull = bool(np.isfinite(occupancy_ratio) and occupancy_ratio >= args.overfull_ratio)

        sample_dir = samples_dir / f"{index:04d}_{sha}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        pred_ply = sample_dir / "pred.ply"
        gt_ply = sample_dir / "gt.ply"
        cond_png = sample_dir / "cond.png"
        write_occupied_points_ply(pred_ply, pred)
        shutil.copy2(gt_path, gt_ply)
        image.save(cond_png)

        reconstruction_metrics = None
        reconstruction_ply = None
        if encoder is not None:
            gt_batch = gt.unsqueeze(0).float().to(device)
            latent = encoder(gt_batch, sample_posterior=False)
            reconstruction_logits = pipeline.models["sparse_structure_decoder"](latent)
            reconstruction = (reconstruction_logits[0] > 0).cpu()
            reconstruction_metrics = compute_binary_metrics(reconstruction, gt)
            reconstruction_ply = sample_dir / "encoder_decoder.ply"
            write_occupied_points_ply(reconstruction_ply, reconstruction)

            reconstruction_count = int(reconstruction_metrics["predicted_occupied_voxels"])
            reconstruction_ratio = reconstruction_metrics["occupancy_ratio"]
            reconstruction_rows.append(
                {
                    "sample_index": index,
                    "sha256": sha,
                    "is_empty_pred": reconstruction_count <= args.empty_threshold,
                    "is_overfull_pred": bool(
                        np.isfinite(reconstruction_ratio) and reconstruction_ratio >= args.overfull_ratio
                    ),
                    **reconstruction_metrics,
                }
            )
            gap_rows.append(
                {
                    "sample_index": index,
                    "sha256": sha,
                    "iou": float(metrics["iou"] - reconstruction_metrics["iou"]),
                    "dice_f1": float(metrics["dice_f1"] - reconstruction_metrics["dice_f1"]),
                    "occupancy_ratio_absolute_error": float(
                        abs(metrics["occupancy_ratio"] - 1.0)
                        - abs(reconstruction_metrics["occupancy_ratio"] - 1.0)
                    ),
                }
            )

        row = {
            "sample_index": index,
            "sha256": sha,
            "seed": sample_seed,
            "condition_image": str(cond_image_path),
            "pred_ply": str(pred_ply),
            "gt_ply": str(gt_ply),
            "cond_png": str(cond_png),
            "is_empty_pred": bool(is_empty),
            "is_overfull_pred": bool(is_overfull),
            **metrics,
        }
        if reconstruction_metrics is not None:
            row.update({
                "encoder_decoder_ply": str(reconstruction_ply),
                **{f"encoder_decoder_{name}": value for name, value in reconstruction_metrics.items()},
                "flow_minus_encoder_decoder_iou": metrics["iou"] - reconstruction_metrics["iou"],
                "flow_minus_encoder_decoder_dice_f1": metrics["dice_f1"] - reconstruction_metrics["dice_f1"],
            })
        rows.append(row)
        print(
            f"[{index + 1}/{len(samples)}] {sha} "
            f"pred={pred_count} gt={gt_count} ratio={occupancy_ratio:.4g} "
            f"iou={metrics['iou']:.4g} dice={metrics['dice_f1']:.4g}",
            flush=True,
        )

    summary = summarize(rows, empty_threshold=args.empty_threshold, overfull_ratio=args.overfull_ratio)
    write_rows_csv(output_dir / "per_sample_metrics.csv", rows)
    write_summary_csv(output_dir / "summary.csv", summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False, allow_nan=True)
    if reconstruction_rows:
        reconstruction_summary = summarize(
            reconstruction_rows,
            empty_threshold=args.empty_threshold,
            overfull_ratio=args.overfull_ratio,
        )
        gap_summary = summarize_gaps(gap_rows)
        write_rows_csv(output_dir / "encoder_decoder_per_sample_metrics.csv", reconstruction_rows)
        with (output_dir / "encoder_decoder_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(reconstruction_summary, fp, indent=2, ensure_ascii=False, allow_nan=True)
        with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "flow_vs_gt": summary,
                    "encoder_decoder_vs_gt": reconstruction_summary,
                    "flow_minus_encoder_decoder": gap_summary,
                },
                fp,
                indent=2,
                ensure_ascii=False,
                allow_nan=True,
            )
        write_rows_csv(
            output_dir / "comparison_summary.csv",
            [
                {
                    "method": "flow_vs_gt",
                    "iou_mean": summary["iou"]["mean"],
                    "dice_f1_mean": summary["dice_f1"]["mean"],
                    "occupancy_ratio_mean": summary["occupancy_ratio"]["mean"],
                },
                {
                    "method": "encoder_decoder_vs_gt",
                    "iou_mean": reconstruction_summary["iou"]["mean"],
                    "dice_f1_mean": reconstruction_summary["dice_f1"]["mean"],
                    "occupancy_ratio_mean": reconstruction_summary["occupancy_ratio"]["mean"],
                },
                {
                    "method": "flow_minus_encoder_decoder",
                    "iou_mean": gap_summary["iou"]["mean"],
                    "dice_f1_mean": gap_summary["dice_f1"]["mean"],
                    "occupancy_ratio_mean": gap_summary["occupancy_ratio_absolute_error"]["mean"],
                },
            ],
        )
    with (output_dir / "eval_config.json").open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2, ensure_ascii=False)
    print(f"Wrote metrics to {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="datasets/Facescape/test", help="Dataset root with metadata.csv, voxels/, renders_cond/")
    parser.add_argument("--output_dir", default="outputs/eval/ss_flow_kl1e-4_step1000", help="Evaluation output directory")
    parser.add_argument("--pipeline_path", default="microsoft/TRELLIS-image-large", help="Base image-to-3D pipeline path")
    parser.add_argument(
        "--flow_ckpt",
        default="outputs/train/ss_flow_finetune_kl1e-4_step1000/ckpts/denoiser_step0001000.pt",
        help="SS flow denoiser checkpoint to evaluate",
    )
    parser.add_argument(
        "--decoder_ckpt",
        default="outputs/train/ss_enc_dec_fine_tune_kl1e-4/ckpts/decoder_step0001000.pt",
        help="Fine-tuned SS decoder checkpoint",
    )
    parser.add_argument("--encoder_config", default=None, help="SS VAE config used to construct the comparison encoder")
    parser.add_argument("--encoder_ckpt", default=None, help="Fine-tuned SS encoder checkpoint for GT reconstruction baseline")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of fixed test samples to evaluate; <=0 means all valid samples")
    parser.add_argument("--resolution", type=int, default=64, help="Dense voxel resolution for GT and sampled sparse structure")
    parser.add_argument("--seed", type=int, default=20260720, help="Seed for sample selection and per-sample flow noise")
    parser.add_argument("--min_aesthetic_score", type=float, default=None, help="Optional metadata aesthetic score filter")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device")
    parser.add_argument("--steps", type=int, default=None, help="Override sparse structure sampler steps")
    parser.add_argument("--cfg_strength", type=float, default=None, help="Override sparse structure CFG strength")
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=None, help="Override sparse structure CFG interval")
    parser.add_argument("--rescale_t", type=float, default=None, help="Override sparse structure sampler rescale_t")
    parser.add_argument("--empty_threshold", type=int, default=0, help="Predicted occupied voxel count at or below this is empty")
    parser.add_argument("--overfull_ratio", type=float, default=3.0, help="Pred/GT occupancy ratio at or above this is overfull")
    parser.add_argument("--no_preprocess_image", dest="preprocess_image", action="store_false", help="Use resized raw condition image")
    parser.set_defaults(preprocess_image=True)
    return parser


def main() -> None:
    evaluate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
