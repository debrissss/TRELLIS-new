"""Comprehensive SparseStructure VAE evaluation for comparing KL weights.

The evaluation covers deterministic reconstruction, sampled-posterior
reconstruction, posterior KL, latent usage/collapse, aggregate-posterior
matching, and standard-normal prior decoding.  It intentionally does not
involve the sparse-structure flow model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate_ss_enc_dec_reconstruction import (
    build_dataset,
    build_model_pair,
    load_checkpoint_manifest,
    load_checkpoints,
    load_config,
    safe_output_name,
    write_occupied_points_ply,
)


RECONSTRUCTION_METRICS = (
    "iou",
    "dice_f1",
    "occupancy_ratio",
    "voxel_error_rate",
    "false_positive_rate",
    "false_negative_rate",
    "bce_with_logits",
    "soft_dice_loss",
)
SURFACE_METRICS = ("chamfer_distance", "average_surface_distance", "hd95")


def _safe_ratio(numerator: float, denominator: float, empty_value: float = float("nan")) -> float:
    return empty_value if denominator == 0 else float(numerator / denominator)


def compute_reconstruction_metrics(
    logits: torch.Tensor,
    gt: torch.Tensor,
    include_surface: bool = False,
) -> dict[str, float | int]:
    """Compute numerically stable voxel reconstruction metrics for one sample."""
    logits64 = logits.detach().double()
    gt_bool = gt.detach().bool()
    gt64 = gt_bool.double()
    pred = logits64 > 0

    intersection = torch.logical_and(pred, gt_bool).sum().item()
    union = torch.logical_or(pred, gt_bool).sum().item()
    pred_count = pred.sum().item()
    gt_count = gt_bool.sum().item()
    false_positive = torch.logical_and(pred, ~gt_bool).sum().item()
    false_negative = torch.logical_and(~pred, gt_bool).sum().item()
    total = gt_bool.numel()
    negative_count = total - gt_count

    probabilities = torch.sigmoid(logits64)
    soft_dice = (2 * (probabilities * gt64).sum() + 1) / (probabilities.sum() + gt64.sum() + 1)
    signed_logits = torch.where(gt_bool, logits64, -logits64)
    bce = F.softplus(-signed_logits).mean()

    metrics: dict[str, float | int] = {
        "iou": 1.0 if union == 0 else float(intersection / union),
        "dice_f1": 1.0 if pred_count + gt_count == 0 else float(2 * intersection / (pred_count + gt_count)),
        "occupancy_ratio": _safe_ratio(pred_count, gt_count),
        "voxel_error_rate": float((false_positive + false_negative) / total),
        "false_positive_rate": _safe_ratio(false_positive, negative_count, empty_value=0.0),
        "false_negative_rate": _safe_ratio(false_negative, gt_count, empty_value=0.0),
        "bce_with_logits": float(bce.item()),
        "soft_dice_loss": float((1 - soft_dice).item()),
        "gt_occupied_voxels": int(gt_count),
        "predicted_occupied_voxels": int(pred_count),
        "intersection_voxels": int(intersection),
        "union_voxels": int(union),
        "false_positive_voxels": int(false_positive),
        "false_negative_voxels": int(false_negative),
        "error_voxels": int(false_positive + false_negative),
    }
    if include_surface:
        metrics.update(compute_surface_metrics(pred, gt_bool))
    return metrics


def _surface_points(occupancy: np.ndarray) -> np.ndarray:
    occupancy = np.asarray(occupancy, dtype=bool).squeeze()
    if occupancy.ndim != 3:
        raise ValueError(f"Expected 3D occupancy, got {occupancy.shape}")
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(occupancy, structure=structure, border_value=0)
    surface = occupancy & ~eroded
    return np.argwhere(surface).astype(np.float64)


def compute_surface_metrics(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Compute symmetric surface distances in normalized voxel coordinates."""
    pred_np = np.asarray(pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred, dtype=bool).squeeze()
    gt_np = np.asarray(gt.detach().cpu() if isinstance(gt, torch.Tensor) else gt, dtype=bool).squeeze()
    if pred_np.shape != gt_np.shape or pred_np.ndim != 3:
        raise ValueError(f"Expected matching 3D occupancies, got {pred_np.shape} and {gt_np.shape}")
    if np.array_equal(pred_np, gt_np):
        return {name: 0.0 for name in SURFACE_METRICS}

    pred_points = _surface_points(pred_np)
    gt_points = _surface_points(gt_np)
    if len(pred_points) == 0 and len(gt_points) == 0:
        return {name: 0.0 for name in SURFACE_METRICS}
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {name: float("inf") for name in SURFACE_METRICS}

    scale = float(max(pred_np.shape))
    pred_to_gt = cKDTree(gt_points).query(pred_points, workers=-1)[0] / scale
    gt_to_pred = cKDTree(pred_points).query(gt_points, workers=-1)[0] / scale
    combined = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "chamfer_distance": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "average_surface_distance": float(combined.mean()),
        "hd95": float(np.percentile(combined, 95)),
    }


def pairwise_dice(predictions: torch.Tensor) -> float:
    """Mean pairwise Dice for [draw, ...] boolean predictions."""
    predictions = predictions.bool().flatten(1)
    if predictions.shape[0] < 2:
        return 1.0
    values = []
    for left in range(predictions.shape[0]):
        for right in range(left + 1, predictions.shape[0]):
            a, b = predictions[left], predictions[right]
            denominator = a.sum().item() + b.sum().item()
            intersection = torch.logical_and(a, b).sum().item()
            values.append(1.0 if denominator == 0 else 2 * intersection / denominator)
    return float(np.mean(values))


def compute_structural_metrics(occupancy: torch.Tensor, dense_threshold: float) -> dict[str, float | int | bool]:
    occupancy_np = np.asarray(occupancy.detach().cpu(), dtype=bool).squeeze()
    occupied = int(occupancy_np.sum())
    total = int(occupancy_np.size)
    fraction = occupied / total
    structure = ndimage.generate_binary_structure(3, 3)
    labels, component_count = ndimage.label(occupancy_np, structure=structure)
    if component_count:
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max())
    else:
        largest = 0
    return {
        "occupied_voxels": occupied,
        "occupancy_fraction": float(fraction),
        "is_empty": bool(occupied == 0),
        "is_dense": bool(fraction > dense_threshold),
        "connected_components": int(component_count),
        "largest_component_voxels": largest,
        "largest_component_ratio": _safe_ratio(largest, occupied, empty_value=0.0),
        "boundary_voxels": int(len(_surface_points(occupancy_np))),
    }


def compute_latent_statistics(
    means: torch.Tensor,
    logvars: torch.Tensor,
    active_threshold: float,
    collapse_threshold: float,
    swd_projections: int = 0,
    swd_max_points: int = 100_000,
    seed: int = 0,
    bootstrap_samples: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize q(z|x) tensors shaped [sample, channel, spatial...]."""
    means = means.double()
    logvars = logvars.double()
    variances = torch.exp(logvars)
    kl_elements = 0.5 * (means.square() + variances - logvars - 1)

    sample_dims = tuple(range(1, means.ndim))
    sample_rows = []
    for index in range(means.shape[0]):
        sigma = torch.exp(0.5 * logvars[index])
        sample_rows.append({
            "dataset_index": index,
            "raw_kl": float(kl_elements[index].mean()),
            "mean_abs": float(means[index].abs().mean()),
            "mean_rms": float(means[index].square().mean().sqrt()),
            "logvar_mean": float(logvars[index].mean()),
            "logvar_min": float(logvars[index].min()),
            "logvar_max": float(logvars[index].max()),
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "sigma_max": float(sigma.max()),
        })

    variance_of_mean = means.var(dim=0, unbiased=False)
    mean_kl_by_element = kl_elements.mean(dim=0)
    active_mask = variance_of_mean > active_threshold
    collapsed_mask = mean_kl_by_element < collapse_threshold
    mean_variance_by_channel = variance_of_mean.flatten(1).mean(dim=1)
    active_channel_mask = mean_variance_by_channel > active_threshold

    channel_rows = []
    channel_reduce_dims = (0,) + tuple(range(2, means.ndim))
    aggregate_mean = means.mean(dim=channel_reduce_dims)
    aggregate_second = (variances + means.square()).mean(dim=channel_reduce_dims)
    aggregate_variance = torch.clamp(aggregate_second - aggregate_mean.square(), min=0)
    for channel in range(means.shape[1]):
        channel_rows.append({
            "channel": channel,
            "raw_kl": float(kl_elements[:, channel].mean()),
            "aggregate_mean": float(aggregate_mean[channel]),
            "aggregate_variance": float(aggregate_variance[channel]),
            "aggregate_std": float(aggregate_variance[channel].sqrt()),
            "active_ratio": float(active_mask[channel].double().mean()),
            "collapse_ratio": float(collapsed_mask[channel].double().mean()),
            "mean_variance_across_samples": float(mean_variance_by_channel[channel]),
            "is_active_channel": bool(active_channel_mask[channel]),
        })

    flattened_means = means.movedim(1, -1).reshape(-1, means.shape[1])
    centered = flattened_means - flattened_means.mean(dim=0)
    covariance = centered.T @ centered / max(flattened_means.shape[0], 1)
    covariance = covariance + torch.diag(variances.movedim(1, -1).reshape(-1, means.shape[1]).mean(dim=0))
    std = torch.diag(covariance).clamp_min(1e-30).sqrt()
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = ~torch.eye(correlation.shape[0], dtype=torch.bool)

    overall = {
        "raw_kl_mean": float(kl_elements.mean()),
        "raw_kl_median_per_sample": float(torch.tensor([r["raw_kl"] for r in sample_rows]).median()),
        "raw_kl_per_sample": summarize_values(
            (row["raw_kl"] for row in sample_rows), bootstrap_samples, seed + 1
        ),
        "active_ratio": float(active_mask.double().mean()),
        "active_elements": int(active_mask.sum()),
        "total_latent_elements": int(active_mask.numel()),
        "active_channels": int(active_channel_mask.sum()),
        "total_latent_channels": int(active_channel_mask.numel()),
        "collapse_ratio": float(collapsed_mask.double().mean()),
        "aggregate_mean_abs_deviation": float(aggregate_mean.abs().mean()),
        "aggregate_variance_abs_deviation": float((aggregate_variance - 1).abs().mean()),
        "aggregate_off_diagonal_correlation_abs_mean": float(correlation[off_diagonal].abs().mean()) if off_diagonal.any() else 0.0,
    }
    if swd_projections > 0:
        overall["sliced_wasserstein_to_standard_normal"] = sliced_wasserstein_to_standard_normal(
            means, logvars, swd_projections, swd_max_points, seed
        )
    return overall, sample_rows, channel_rows


def sliced_wasserstein_to_standard_normal(
    means: torch.Tensor,
    logvars: torch.Tensor,
    projections: int,
    max_points: int,
    seed: int,
) -> float:
    """Compare the channel-wise aggregate posterior with N(0, I)."""
    if projections <= 0 or max_points <= 0:
        raise ValueError("projections and max_points must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flattened_mean = means.movedim(1, -1).reshape(-1, means.shape[1]).float().cpu()
    flattened_logvar = logvars.movedim(1, -1).reshape(-1, means.shape[1]).float().cpu()
    if flattened_mean.shape[0] > max_points:
        indices = torch.randperm(flattened_mean.shape[0], generator=generator)[:max_points]
        flattened_mean = flattened_mean[indices]
        flattened_logvar = flattened_logvar[indices]
    epsilon = torch.randn(flattened_mean.shape, generator=generator)
    posterior = flattened_mean + torch.exp(0.5 * flattened_logvar) * epsilon
    reference = torch.randn(posterior.shape, generator=generator)
    directions = torch.randn((projections, posterior.shape[1]), generator=generator)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    posterior_projection = torch.sort(posterior @ directions.T, dim=0).values
    reference_projection = torch.sort(reference @ directions.T, dim=0).values
    return float((posterior_projection - reference_projection).abs().mean())


def summarize_values(values: Iterable[float], bootstrap_samples: int = 0, seed: int = 0) -> dict[str, float | int]:
    values_np = np.asarray(list(values), dtype=np.float64)
    finite = values_np[np.isfinite(values_np)]
    if finite.size == 0:
        return {key: float("nan") for key in ("mean", "median", "std", "p10", "p90", "min", "max", "ci95_low", "ci95_high")} | {"count": 0}
    result: dict[str, float | int] = {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
    if bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, finite.size, size=(bootstrap_samples, finite.size))
        bootstrap_means = finite[indices].mean(axis=1)
        result["ci95_low"] = float(np.percentile(bootstrap_means, 2.5))
        result["ci95_high"] = float(np.percentile(bootstrap_means, 97.5))
    else:
        result["ci95_low"] = float("nan")
        result["ci95_high"] = float("nan")
    return result


def _summarize_rows(rows: list[dict[str, Any]], metric_names: Iterable[str], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    return {
        name: summarize_values((float(row[name]) for row in rows), bootstrap_samples, seed + offset)
        for offset, name in enumerate(metric_names)
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_batches(dataset, batch_size: int, num_samples: int):
    limit = len(dataset) if num_samples <= 0 else min(num_samples, len(dataset))
    for start in range(0, limit, batch_size):
        indices = list(range(start, min(start + batch_size, limit)))
        yield indices, torch.stack([dataset[index]["ss"] for index in indices], dim=0)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_name: str,
    checkpoint_entry: dict[str, Any],
    model_pair: dict[str, torch.nn.Module],
    dataset,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    device = torch.device(args.device)
    encoder, decoder = model_pair["encoder"], model_pair["decoder"]
    checkpoint_dir = Path(args.output_dir) / "samples" / safe_output_name(checkpoint_name)
    lambda_kl = checkpoint_entry.get("lambda_kl")
    posterior_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    mean_rows: list[dict[str, Any]] = []
    posterior_draw_rows: list[dict[str, Any]] = []
    posterior_sample_rows: list[dict[str, Any]] = []
    all_means, all_logvars = [], []
    latent_metadata: list[tuple[str, str]] = []
    gt_counts: list[int] = []
    latent_shape: tuple[int, ...] | None = None

    for indices, gt_cpu in _dataset_batches(dataset, args.batch_size, args.num_samples):
        gt = gt_cpu.to(device).float()
        z_mean, mean, logvar = encoder(gt, sample_posterior=False, return_raw=True)
        mean_logits = decoder(z_mean)
        all_means.append(mean.cpu())
        all_logvars.append(logvar.cpu())
        latent_shape = tuple(mean.shape[1:])

        batch_draw_predictions: list[torch.Tensor] = []
        batch_draw_ious = np.empty((args.posterior_draws, len(indices)), dtype=np.float64)
        first_predictions: torch.Tensor | None = None
        for draw in range(args.posterior_draws):
            epsilon = torch.randn(mean.shape, generator=posterior_generator, dtype=mean.dtype).to(device)
            sampled_logits = decoder(mean + torch.exp(0.5 * logvar) * epsilon)
            draw_predictions = (sampled_logits > 0).cpu()
            batch_draw_predictions.append(draw_predictions)
            if first_predictions is None:
                first_predictions = draw_predictions
            for local_index, dataset_index in enumerate(indices):
                root, sha = dataset.instances[dataset_index]
                metrics = compute_reconstruction_metrics(sampled_logits[local_index].cpu(), gt_cpu[local_index])
                batch_draw_ious[draw, local_index] = metrics["iou"]
                posterior_draw_rows.append({
                    "checkpoint": checkpoint_name,
                    "dataset_index": dataset_index,
                    "root": root,
                    "sha256": sha,
                    "draw": draw,
                    **metrics,
                })

        stacked_predictions = torch.stack(batch_draw_predictions, dim=0)
        for local_index, dataset_index in enumerate(indices):
            root, sha = dataset.instances[dataset_index]
            latent_metadata.append((root, sha))
            mean_metrics = compute_reconstruction_metrics(mean_logits[local_index].cpu(), gt_cpu[local_index], include_surface=not args.skip_surface_metrics)
            gt_counts.append(int(mean_metrics["gt_occupied_voxels"]))
            mean_row = {
                "checkpoint": checkpoint_name,
                "dataset_index": dataset_index,
                "root": root,
                "sha256": sha,
                **mean_metrics,
            }
            sample_draw_rows = [row for row in posterior_draw_rows if row["dataset_index"] == dataset_index and row["checkpoint"] == checkpoint_name]
            posterior_sample_rows.append({
                "checkpoint": checkpoint_name,
                "dataset_index": dataset_index,
                "root": root,
                "sha256": sha,
                "iou_mean": float(np.mean([row["iou"] for row in sample_draw_rows])),
                "iou_std": float(np.std([row["iou"] for row in sample_draw_rows])),
                "iou_min": float(np.min([row["iou"] for row in sample_draw_rows])),
                "dice_f1_mean": float(np.mean([row["dice_f1"] for row in sample_draw_rows])),
                "bce_with_logits_mean": float(np.mean([row["bce_with_logits"] for row in sample_draw_rows])),
                "soft_dice_loss_mean": float(np.mean([row["soft_dice_loss"] for row in sample_draw_rows])),
                "error_voxels_mean": float(np.mean([row["error_voxels"] for row in sample_draw_rows])),
                "perfect_reconstruction_rate": float(np.mean([row["error_voxels"] == 0 for row in sample_draw_rows])),
                "pairwise_dice": pairwise_dice(stacked_predictions[:, local_index]),
            })

            if args.export_ply:
                sample_dir = checkpoint_dir / "posterior_mean" / f"{dataset_index:04d}_{sha}"
                gt_path = sample_dir / "gt.ply"
                pred_path = sample_dir / "pred.ply"
                write_occupied_points_ply(gt_path, gt_cpu[local_index])
                write_occupied_points_ply(pred_path, mean_logits[local_index].cpu() > 0)
                mean_row["gt_ply"] = str(gt_path)
                mean_row["pred_ply"] = str(pred_path)

                draw_indices = list(range(args.posterior_draws)) if args.export_all_posterior_draws else sorted({0, int(batch_draw_ious[:, local_index].argmin())})
                posterior_dir = checkpoint_dir / "posterior_sample" / f"{dataset_index:04d}_{sha}"
                for draw in draw_indices:
                    write_occupied_points_ply(posterior_dir / f"draw_{draw:03d}.ply", stacked_predictions[draw, local_index])
            mean_rows.append(mean_row)

        del gt, z_mean, mean, logvar, mean_logits, sampled_logits

    means = torch.cat(all_means, dim=0)
    logvars = torch.cat(all_logvars, dim=0)
    latent_overall, latent_rows, channel_rows = compute_latent_statistics(
        means,
        logvars,
        args.active_threshold,
        args.collapse_threshold,
        args.swd_projections,
        args.swd_max_points,
        args.seed + 30_000,
        args.bootstrap_samples,
    )
    for index, row in enumerate(latent_rows):
        root, sha = latent_metadata[index]
        row.update({
            "checkpoint": checkpoint_name,
            "root": root,
            "sha256": sha,
            "lambda_kl": lambda_kl,
            "weighted_kl": None if lambda_kl is None else float(lambda_kl * row["raw_kl"]),
        })
    for row in channel_rows:
        row.update({"checkpoint": checkpoint_name, "lambda_kl": lambda_kl})
    latent_overall["lambda_kl"] = lambda_kl
    latent_overall["weighted_kl_mean"] = None if lambda_kl is None else float(lambda_kl * latent_overall["raw_kl_mean"])

    if latent_shape is None:
        raise RuntimeError("No latent tensors were evaluated")
    prior_rows = []
    prior_generator = torch.Generator(device="cpu").manual_seed(args.seed + 100_000)
    for start in range(0, args.prior_draws, args.batch_size):
        batch_count = min(args.batch_size, args.prior_draws - start)
        prior = torch.randn((batch_count, *latent_shape), generator=prior_generator).to(device)
        prior_logits = decoder(prior)
        for local_index in range(batch_count):
            prior_index = start + local_index
            occupancy = prior_logits[local_index].cpu() > 0
            row = {
                "checkpoint": checkpoint_name,
                "prior_index": prior_index,
                **compute_structural_metrics(occupancy, args.dense_occupancy_threshold),
            }
            if args.export_ply:
                ply_path = checkpoint_dir / "prior" / f"prior_{prior_index:04d}.ply"
                write_occupied_points_ply(ply_path, occupancy)
                row["pred_ply"] = str(ply_path)
            prior_rows.append(row)
        del prior, prior_logits

    sample_summary_metrics = (
        "iou_mean", "iou_std", "iou_min", "dice_f1_mean", "bce_with_logits_mean",
        "soft_dice_loss_mean", "error_voxels_mean", "perfect_reconstruction_rate", "pairwise_dice",
    )
    prior_numeric_metrics = (
        "occupied_voxels", "occupancy_fraction", "connected_components",
        "largest_component_ratio", "boundary_voxels",
    )
    summary = {
        "num_samples": len(mean_rows),
        "posterior_draws": args.posterior_draws,
        "prior_draws": args.prior_draws,
        "posterior_mean_reconstruction": _summarize_rows(
            mean_rows,
            RECONSTRUCTION_METRICS + (() if args.skip_surface_metrics else SURFACE_METRICS),
            args.bootstrap_samples,
            args.seed,
        ),
        "sampled_posterior_reconstruction": _summarize_rows(
            posterior_sample_rows, sample_summary_metrics, args.bootstrap_samples, args.seed + 10_000
        ),
        "posterior_kl_and_latent": latent_overall,
        "prior_decoding": {
            **_summarize_rows(prior_rows, prior_numeric_metrics, args.bootstrap_samples, args.seed + 20_000),
            "empty_rate": float(np.mean([row["is_empty"] for row in prior_rows])),
            "dense_rate": float(np.mean([row["is_dense"] for row in prior_rows])),
            "occupied_voxel_wasserstein_to_test": float(wasserstein_distance(
                [row["occupied_voxels"] for row in prior_rows], gt_counts
            )),
        },
    }
    rows_by_type = {
        "posterior_mean": mean_rows,
        "posterior_draw": posterior_draw_rows,
        "posterior_sample": posterior_sample_rows,
        "latent_sample": latent_rows,
        "latent_channel": channel_rows,
        "prior_sample": prior_rows,
    }
    return summary, rows_by_type


def _flatten_summary(checkpoint: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"checkpoint": checkpoint}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_name, child_value in value.items():
                visit(f"{prefix}.{child_name}" if prefix else child_name, child_value)
        else:
            row[prefix] = value

    visit("", summary)
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_names", nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=0, help="0 evaluates the full provided dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--posterior_draws", type=int, default=10)
    parser.add_argument("--prior_draws", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--active_threshold", type=float, default=1e-2)
    parser.add_argument("--collapse_threshold", type=float, default=1e-3)
    parser.add_argument("--dense_occupancy_threshold", type=float, default=0.25)
    parser.add_argument("--swd_projections", type=int, default=0, help="0 disables aggregate-posterior SWD")
    parser.add_argument("--swd_max_points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_surface_metrics", action="store_true")
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--export_all_posterior_draws", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.batch_size <= 0 or args.posterior_draws <= 0 or args.prior_draws <= 0:
        raise ValueError("batch_size, posterior_draws, and prior_draws must be positive")
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.data_root)
    manifest = load_checkpoint_manifest(args.checkpoints)
    if args.checkpoint_names:
        missing = sorted(set(args.checkpoint_names) - set(manifest))
        if missing:
            raise KeyError(f"Unknown checkpoint names: {missing}")
        manifest = {name: manifest[name] for name in args.checkpoint_names}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {
        "posterior_mean": [], "posterior_draw": [], "posterior_sample": [],
        "latent_sample": [], "latent_channel": [], "prior_sample": [],
    }
    summaries: dict[str, Any] = {}
    checkpoint_metadata: dict[str, Any] = {
        "config": args.config,
        "data_root": args.data_root,
        "seed": args.seed,
        "posterior_draws": args.posterior_draws,
        "prior_draws": args.prior_draws,
        "active_threshold": args.active_threshold,
        "collapse_threshold": args.collapse_threshold,
        "swd_projections": args.swd_projections,
        "swd_max_points": args.swd_max_points,
        "checkpoints": {},
    }

    device = torch.device(args.device)
    for checkpoint_name, entry in manifest.items():
        print(f"Evaluating {checkpoint_name} on {min(len(dataset), args.num_samples or len(dataset))} samples", flush=True)
        models = build_model_pair(cfg, device)
        load_checkpoints(models, entry)
        summary, rows_by_type = evaluate_checkpoint(checkpoint_name, entry, models, dataset, args)
        summaries[checkpoint_name] = summary
        for name, rows in rows_by_type.items():
            all_rows[name].extend(rows)
        checkpoint_metadata["checkpoints"][checkpoint_name] = {
            "lambda_kl": entry.get("lambda_kl"),
            "encoder": entry["encoder"],
            "decoder": entry["decoder"],
            "encoder_sha256": _sha256(entry["encoder"]),
            "decoder_sha256": _sha256(entry["decoder"]),
        }
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output_dir / "posterior_mean_per_sample.csv", all_rows["posterior_mean"])
    _write_csv(output_dir / "posterior_sample_per_draw.csv", all_rows["posterior_draw"])
    _write_csv(output_dir / "posterior_sample_per_sample.csv", all_rows["posterior_sample"])
    _write_csv(output_dir / "latent_per_sample.csv", all_rows["latent_sample"])
    _write_csv(output_dir / "latent_per_channel.csv", all_rows["latent_channel"])
    _write_csv(output_dir / "prior_sample_metrics.csv", all_rows["prior_sample"])
    _write_csv(output_dir / "summary.csv", [_flatten_summary(name, summary) for name, summary in summaries.items()])
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2, allow_nan=True)
    with open(output_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(checkpoint_metadata, fp, indent=2)
    print(f"Wrote comprehensive evaluation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
