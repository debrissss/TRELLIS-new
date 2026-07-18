"""Evaluate SparseStructure encoder/decoder reconstruction on a fixed dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import datasets, models


METRIC_NAMES = ("iou", "dice_f1", "occupancy_ratio", "soft_dice_loss")


def compute_binary_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float | int]:
    pred = pred.bool()
    gt = gt.bool()
    intersection = torch.logical_and(pred, gt).sum().item()
    union = torch.logical_or(pred, gt).sum().item()
    pred_count = pred.sum().item()
    gt_count = gt.sum().item()

    iou = 1.0 if union == 0 else intersection / union
    dice_f1 = 1.0 if pred_count + gt_count == 0 else (2 * intersection) / (pred_count + gt_count)
    occupancy_ratio = float("nan") if gt_count == 0 else pred_count / gt_count

    return {
        "iou": float(iou),
        "dice_f1": float(dice_f1),
        "occupancy_ratio": float(occupancy_ratio),
        "gt_occupied_voxels": int(gt_count),
        "predicted_occupied_voxels": int(pred_count),
        "intersection_voxels": int(intersection),
        "union_voxels": int(union),
    }


def compute_reconstruction_metrics(logits: torch.Tensor, gt: torch.Tensor) -> dict[str, float | int]:
    gt_float = gt.float()
    probs = torch.sigmoid(logits.float())
    soft_dice_score = (2 * (probs * gt_float).sum() + 1) / (probs.sum() + gt_float.sum() + 1)
    metrics = compute_binary_metrics(logits > 0, gt.bool())
    metrics["soft_dice_loss"] = float((1 - soft_dice_score).item())
    return metrics


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[metric] = {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "p10": float("nan"),
                "p90": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
            continue
        summary[metric] = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p90": float(np.percentile(finite, 90)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    for count_name in ("gt_occupied_voxels", "predicted_occupied_voxels"):
        values = np.asarray([row[count_name] for row in rows], dtype=np.float64)
        summary[count_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }
    return summary


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_config(path: str | Path) -> edict:
    return edict(load_json(path))


def build_model_pair(cfg: edict, device: torch.device) -> dict[str, torch.nn.Module]:
    model_dict = {}
    for name in ("encoder", "decoder"):
        if name not in cfg.models:
            raise KeyError(f"Config is missing models.{name}")
        model_cfg = cfg.models[name]
        model = getattr(models, model_cfg.name)(**model_cfg.args).to(device)
        if device.type == "cpu" and hasattr(model, "convert_to_fp32"):
            model.convert_to_fp32()
        model.eval()
        model_dict[name] = model
    return model_dict


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    return state


def load_checkpoints(models_by_name: dict[str, torch.nn.Module], checkpoint_paths: dict[str, str]) -> None:
    for model_name in ("encoder", "decoder"):
        if model_name not in checkpoint_paths:
            raise KeyError(f"Checkpoint entry is missing {model_name}")
        state = _load_state_dict(checkpoint_paths[model_name])
        models_by_name[model_name].load_state_dict(state)
        models_by_name[model_name].eval()


def load_checkpoint_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    manifest = load_json(path)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Checkpoint manifest must be a non-empty object")
    for name, entry in manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Checkpoint entry must be an object: {name}")
        for key in ("encoder", "decoder"):
            if key not in entry:
                raise ValueError(f"Checkpoint entry {name} is missing {key}")
    return manifest


def build_dataset(cfg: edict, data_root: str):
    dataset_cfg = cfg.dataset
    return getattr(datasets, dataset_cfg.name)(data_root, **dataset_cfg.args)


def _batched_indices(num_items: int, batch_size: int):
    for start in range(0, num_items, batch_size):
        yield list(range(start, min(start + batch_size, num_items)))


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_name: str,
    model_pair: dict[str, torch.nn.Module],
    dataset,
    batch_size: int,
    device: torch.device,
    sample_posterior: bool = False,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows: list[dict[str, Any]] = []
    encoder = model_pair["encoder"]
    decoder = model_pair["decoder"]

    for indices in _batched_indices(len(dataset), batch_size):
        samples = [dataset[index]["ss"] for index in indices]
        gt_batch = torch.stack(samples, dim=0).to(device)
        z = encoder(gt_batch.float(), sample_posterior=sample_posterior)
        logits = decoder(z)
        for local_index, dataset_index in enumerate(indices):
            root, sha = dataset.instances[dataset_index]
            metrics = compute_reconstruction_metrics(logits[local_index].cpu(), gt_batch[local_index].cpu())
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "mode": "sample_posterior" if sample_posterior else "posterior_mean",
                    "dataset_index": dataset_index,
                    "root": root,
                    "sha256": sha,
                    **metrics,
                }
            )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str | Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for checkpoint, summary in summaries.items():
        flat = {"checkpoint": checkpoint, "num_samples": summary["num_samples"]}
        for metric in METRIC_NAMES:
            for stat_name, value in summary[metric].items():
                flat[f"{metric}_{stat_name}"] = value
        for count_name in ("gt_occupied_voxels", "predicted_occupied_voxels"):
            for stat_name, value in summary[count_name].items():
                flat[f"{count_name}_{stat_name}"] = value
        rows.append(flat)
    write_rows_csv(path, rows)


def safe_output_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="SS VAE config JSON")
    parser.add_argument("--data_root", required=True, help="Fixed mini dataset root with metadata.csv and voxels/")
    parser.add_argument("--checkpoints", required=True, help="JSON checkpoint manifest")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and JSON metrics")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--checkpoint_names", nargs="*", default=None, help="Optional subset of checkpoint names")
    parser.add_argument("--sample_posterior", action="store_true", help="Sample from the posterior instead of using the posterior mean")
    parser.add_argument("--seed", type=int, default=20260718, help="Random seed for posterior sampling")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = load_config(args.config)
    manifest = load_checkpoint_manifest(args.checkpoints)
    if args.checkpoint_names:
        missing = sorted(set(args.checkpoint_names) - set(manifest))
        if missing:
            raise KeyError(f"Unknown checkpoint names: {missing}")
        manifest = {name: manifest[name] for name in args.checkpoint_names}

    device = torch.device(args.device)
    dataset = build_dataset(cfg, args.data_root)
    if len(dataset) == 0:
        raise ValueError(f"Evaluation dataset has no samples: {args.data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}

    for checkpoint_name, checkpoint_paths in manifest.items():
        print(f"Evaluating {checkpoint_name} on {len(dataset)} samples")
        model_pair = build_model_pair(cfg, device=device)
        load_checkpoints(model_pair, checkpoint_paths)
        rows = evaluate_checkpoint(
            checkpoint_name=checkpoint_name,
            model_pair=model_pair,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            sample_posterior=args.sample_posterior,
        )
        write_rows_csv(output_dir / f"{safe_output_name(checkpoint_name)}_per_sample_metrics.csv", rows)
        summaries[checkpoint_name] = summarize_metric_rows(rows)
        del model_pair
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2, allow_nan=True)
    write_summary_csv(output_dir / "summary.csv", summaries)
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
