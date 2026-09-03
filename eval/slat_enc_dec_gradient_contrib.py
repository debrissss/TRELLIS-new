#!/usr/bin/env python3
"""Measure local encoder gradient contribution of KL for SLat enc/dec checkpoints."""

# 中文说明：
# 评估 SLat encoder + GS decoder 训练目标中各项 loss 对 encoder 梯度的局部贡献。
# 重点统计 KL 项相对总梯度的范数比例、能量比例、方向 cosine 和 projection。
# 这是诊断脚本，不负责重建质量排序。

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.impl.slat_encoder_gs_decoder_reconstruction_impl import (  # noqa: E402
    build_models,
    build_renderer,
    l1_metric,
    load_checkpoint,
    load_fixed_view,
    load_json,
    load_sparse_feature,
    lpips_metric,
    parse_view_indices,
    read_metadata_rows,
    render_batch,
    ssim_metric,
)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    lambda_kl: float
    encoder_ckpt: Path
    decoder_ckpt: Path


def parse_checkpoint_spec(text: str) -> CheckpointSpec:
    parts = text.split("=", 3)
    if len(parts) != 4:
        raise ValueError(
            "--checkpoints entries must be NAME=LAMBDA_KL=ENCODER_CKPT=DECODER_CKPT, "
            f"got: {text}"
        )
    name, lambda_kl, encoder_ckpt, decoder_ckpt = parts
    if not name:
        raise ValueError(f"Checkpoint name is empty in spec: {text}")
    return CheckpointSpec(
        name=name,
        lambda_kl=float(lambda_kl),
        encoder_ckpt=Path(encoder_ckpt),
        decoder_ckpt=Path(decoder_ckpt),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regularization_loss(reps, regularizations: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = None
    terms: dict[str, torch.Tensor] = {}
    if "lambda_vol" in regularizations:
        scales = torch.cat([g.get_scaling for g in reps], dim=0)
        terms["reg_vol"] = torch.prod(scales, dim=1).mean()
        loss = regularizations["lambda_vol"] * terms["reg_vol"] if loss is None else loss + regularizations["lambda_vol"] * terms["reg_vol"]
    if "lambda_opacity" in regularizations:
        opacity = torch.cat([g.get_opacity for g in reps], dim=0)
        terms["reg_opacity"] = (opacity - 1).pow(2).mean()
        loss = regularizations["lambda_opacity"] * terms["reg_opacity"] if loss is None else loss + regularizations["lambda_opacity"] * terms["reg_opacity"]
    if loss is None:
        device = reps[0].get_xyz.device if reps else torch.device("cuda")
        loss = torch.zeros((), device=device)
    return loss, terms


def grad_list(scalar: torch.Tensor, params: list[torch.nn.Parameter]) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)


def grad_norm_sq(grads: tuple[torch.Tensor | None, ...], device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().float().pow(2).sum()
    return total


def grad_dot(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for left_grad, right_grad in zip(left, right):
        if left_grad is not None and right_grad is not None:
            total = total + (left_grad.detach().float() * right_grad.detach().float()).sum()
    return total


def safe_float(value: torch.Tensor) -> float:
    value = value.detach().float().cpu()
    if value.numel() != 1:
        raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
    return float(value.item())


def gradient_contribution_stats(
    *,
    rec: torch.Tensor,
    weighted_kl: torch.Tensor,
    total: torch.Tensor,
    params: list[torch.nn.Parameter],
    device: torch.device,
) -> dict[str, float]:
    rec_grads = grad_list(rec, params)
    kl_grads = grad_list(weighted_kl, params)
    total_grads = grad_list(total, params)

    rec_norm_sq = grad_norm_sq(rec_grads, device)
    kl_norm_sq = grad_norm_sq(kl_grads, device)
    total_norm_sq = grad_norm_sq(total_grads, device)
    kl_total_dot = grad_dot(kl_grads, total_grads, device)
    kl_rec_dot = grad_dot(kl_grads, rec_grads, device)
    rec_total_dot = grad_dot(rec_grads, total_grads, device)

    eps = torch.tensor(1e-20, device=device)
    rec_norm = torch.sqrt(rec_norm_sq + eps)
    kl_norm = torch.sqrt(kl_norm_sq + eps)
    total_norm = torch.sqrt(total_norm_sq + eps)

    return {
        "encoder_grad_norm_rec": safe_float(rec_norm),
        "encoder_grad_norm_kl": safe_float(kl_norm),
        "encoder_grad_norm_total": safe_float(total_norm),
        "encoder_grad_ratio_kl_total": safe_float(kl_norm / (total_norm + eps)),
        "encoder_grad_energy_ratio_kl_total": safe_float(kl_norm_sq / (total_norm_sq + eps)),
        "encoder_grad_projection_kl_total": safe_float(kl_total_dot / (total_norm_sq + eps)),
        "encoder_grad_cosine_kl_total": safe_float(kl_total_dot / (kl_norm * total_norm + eps)),
        "encoder_grad_cosine_kl_rec": safe_float(kl_rec_dot / (kl_norm * rec_norm + eps)),
        "encoder_grad_cosine_rec_total": safe_float(rec_total_dot / (rec_norm * total_norm + eps)),
    }


def compute_one_record(
    *,
    model_dict: dict[str, torch.nn.Module],
    renderer,
    data_dir: Path,
    sha: str,
    view_index: int,
    image_size: int,
    feature_model: str,
    resolution: int,
    trainer_args: dict[str, Any],
    lambda_kl: float,
    sample_posterior: bool,
    skip_lpips: bool,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    view = load_fixed_view(data_dir, sha, view_index, image_size)
    feats = load_sparse_feature(data_dir, sha, feature_model, resolution).to(device)
    image = view["image"].unsqueeze(0).to(device)
    alpha = view["alpha"].unsqueeze(0).to(device)
    extrinsics = view["extrinsics"].unsqueeze(0).to(device)
    intrinsics = view["intrinsics"].unsqueeze(0).to(device)

    z, mean, logvar = model_dict["encoder"](feats, sample_posterior=sample_posterior, return_raw=True)
    reps = model_dict["decoder"](z)
    render_results = render_batch(renderer, reps, extrinsics, intrinsics)
    rec_image = render_results["color"]
    bg_color = render_results["bg_color"][..., None, None]
    gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * bg_color

    lambda_ssim = float(trainer_args.get("lambda_ssim", 0.0))
    lambda_lpips = float(trainer_args.get("lambda_lpips", 0.0))
    regularizations = dict(trainer_args.get("regularizations", {}))

    l1 = l1_metric(rec_image, gt_image)
    rec = l1
    ssim_loss = 1 - ssim_metric(rec_image, gt_image)
    rec = rec + lambda_ssim * ssim_loss
    if skip_lpips:
        lpips_loss = torch.full((), math.nan, device=device)
    else:
        lpips_loss = lpips_metric(rec_image, gt_image)
        rec = rec + lambda_lpips * lpips_loss
    kl = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
    weighted_kl = lambda_kl * kl
    reg_loss, reg_terms = regularization_loss(reps, regularizations)
    total = rec + weighted_kl + reg_loss

    encoder_params = [p for p in model_dict["encoder"].parameters() if p.requires_grad]
    grad_stats = gradient_contribution_stats(
        rec=rec,
        weighted_kl=weighted_kl,
        total=total,
        params=encoder_params,
        device=device,
    )

    row = {
        "loss": safe_float(total),
        "rec": safe_float(rec),
        "l1": safe_float(l1),
        "ssim_loss": safe_float(ssim_loss),
        "lpips": safe_float(lpips_loss) if not skip_lpips else math.nan,
        "kl": safe_float(kl),
        "weighted_kl": safe_float(weighted_kl),
        "weighted_kl_loss_ratio": safe_float(weighted_kl / (total.detach() + 1e-20)),
        "reg_loss": safe_float(reg_loss),
    }
    for name, term in reg_terms.items():
        row[name] = safe_float(term)
    row.update(grad_stats)
    return row


def numeric_summary(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row and not math.isnan(float(row[metric]))]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        summary[metric] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std(ddof=0)),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leading = ["run", "lambda_kl", "encoder_ckpt", "decoder_ckpt", "sha256", "view_index", "failed", "error"]
    remaining = sorted({key for row in rows for key in row} - set(leading))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=leading + remaining, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    dataset_args = config["dataset"]["args"]
    trainer_args = dict(config["trainer"]["args"])
    image_size = int(dataset_args["image_size"])
    feature_model = args.feature_model or dataset_args.get("model", "dinov2_vitl14_reg")
    resolution = int(dataset_args.get("resolution", 64))
    view_indices = parse_view_indices(args.view_indices)
    device = torch.device(args.device)
    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoints]
    selected_rows = read_metadata_rows(args.data_dir)[: args.num_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    metric_names = [
        "loss",
        "rec",
        "l1",
        "ssim_loss",
        "lpips",
        "kl",
        "weighted_kl",
        "weighted_kl_loss_ratio",
        "reg_loss",
        "encoder_grad_norm_rec",
        "encoder_grad_norm_kl",
        "encoder_grad_norm_total",
        "encoder_grad_ratio_kl_total",
        "encoder_grad_energy_ratio_kl_total",
        "encoder_grad_projection_kl_total",
        "encoder_grad_cosine_kl_total",
        "encoder_grad_cosine_kl_rec",
        "encoder_grad_cosine_rec_total",
    ]

    for spec in checkpoint_specs:
        model_dict = build_models(config, device, trainer_args.get("fp16_mode"))
        load_checkpoint(model_dict["encoder"], spec.encoder_ckpt, device, trainer_args.get("fp16_mode"))
        load_checkpoint(model_dict["decoder"], spec.decoder_ckpt, device, trainer_args.get("fp16_mode"))
        for param in model_dict["decoder"].parameters():
            param.requires_grad_(False)
        renderer = build_renderer(model_dict["decoder"])
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.resolution = image_size

        for sample_idx, metadata_row in enumerate(selected_rows):
            sha = metadata_row["sha256"]
            for view_index in view_indices:
                base_row: dict[str, Any] = {
                    "run": spec.name,
                    "lambda_kl": spec.lambda_kl,
                    "encoder_ckpt": str(spec.encoder_ckpt),
                    "decoder_ckpt": str(spec.decoder_ckpt),
                    "sha256": sha,
                    "view_index": view_index,
                }
                try:
                    record = compute_one_record(
                        model_dict=model_dict,
                        renderer=renderer,
                        data_dir=args.data_dir,
                        sha=sha,
                        view_index=view_index,
                        image_size=image_size,
                        feature_model=feature_model,
                        resolution=resolution,
                        trainer_args=trainer_args,
                        lambda_kl=spec.lambda_kl,
                        sample_posterior=args.sample_posterior,
                        skip_lpips=args.skip_lpips,
                        seed=args.seed + sample_idx * 1000 + view_index,
                        device=device,
                    )
                    all_rows.append({**base_row, **record, "failed": False, "error": ""})
                except Exception as exc:
                    row = {**base_row, "failed": True, "error": repr(exc)}
                    all_rows.append(row)
                    failed_rows.append(row)
                    if args.fail_on_error:
                        raise
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        del model_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_runs: dict[str, Any] = {}
    for spec in checkpoint_specs:
        run_rows = [row for row in all_rows if row["run"] == spec.name and not row.get("failed")]
        summary_runs[spec.name] = {
            "lambda_kl": spec.lambda_kl,
            "encoder_ckpt": str(spec.encoder_ckpt),
            "decoder_ckpt": str(spec.decoder_ckpt),
            "num_records": len(run_rows),
            "failed_count": len([row for row in all_rows if row["run"] == spec.name and row.get("failed")]),
            "metrics": numeric_summary(run_rows, metric_names),
        }

    summary = {
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "feature_model": feature_model,
        "view_indices": view_indices,
        "num_samples": args.num_samples,
        "sample_posterior": bool(args.sample_posterior),
        "skip_lpips": bool(args.skip_lpips),
        "seed": args.seed,
        "runs": summary_runs,
        "num_records": len([row for row in all_rows if not row.get("failed")]),
        "failed_count": len(failed_rows),
    }

    write_csv(args.output_dir / "per_sample.csv", all_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "failed_samples.json").write_text(json.dumps(failed_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more NAME=LAMBDA_KL=ENCODER_CKPT=DECODER_CKPT entries.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--view_indices", default="0")
    parser.add_argument("--feature_model", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--sample_posterior", action="store_true", help="Use stochastic posterior sampling with a fixed seed.")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS in rec gradient for faster diagnostics.")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
