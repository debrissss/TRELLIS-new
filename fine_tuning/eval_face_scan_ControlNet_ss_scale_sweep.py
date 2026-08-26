"""Compare SS ControlNet timestep schedules on the FaceScan repair task.

The evaluator keeps the input, checkpoint, normal image, prepared 3D control,
base control scale, sampler settings, and random seed fixed. Only the timestep
schedule changes between variants. It deliberately stops after SS decoding;
SLat models and SLat distillation are outside this experiment.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("SPCONV_ALGO", "native")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import trimesh
import utils3d
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file

from trellis import models
from trellis.pipelines import samplers
from trellis.pipelines.trellis_image_to_3d_ControlNet import (
    TrellisImageTo3DPipeline_ControlNet,
)


RESOLUTION = 64
CHECKPOINT_BASENAME = "ss_flow_ControlNet"


def default_schedule_variants(include_progress: bool = False) -> list[dict]:
    variants = [
        {"name": "baseline", "schedule": None},
        {
            "name": "mild",
            "schedule": {
                "name": "smoothstep",
                "domain": "flow_t",
                "full_strength_t": 0.65,
                "min_strength_t": 0.25,
                "min_scale": 0.1,
            },
        },
        {
            "name": "release",
            "schedule": {
                "name": "smoothstep",
                "domain": "flow_t",
                "full_strength_t": 0.65,
                "min_strength_t": 0.25,
                "min_scale": 0.0,
            },
        },
        {
            "name": "earlier_release",
            "schedule": {
                "name": "smoothstep",
                "domain": "flow_t",
                "full_strength_t": 0.8,
                "min_strength_t": 0.4,
                "min_scale": 0.1,
            },
        },
    ]
    if include_progress:
        variants.append({
            "name": "progress_mild",
            "schedule": {
                "name": "smoothstep",
                "domain": "progress",
                "full_until": 0.6,
                "fade_until": 0.85,
                "min_scale": 0.1,
            },
        })
    return variants


def load_schedule_variants(
    variants_json: Optional[Path],
    include_progress: bool,
) -> list[dict]:
    variants = (
        json.loads(variants_json.read_text(encoding="utf-8"))
        if variants_json is not None
        else default_schedule_variants(include_progress=include_progress)
    )
    if not isinstance(variants, list) or not variants:
        raise ValueError("Schedule variants must be a non-empty JSON list")
    names = []
    normalized = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise TypeError(f"Schedule variant {index} must be an object")
        unknown = set(variant) - {"name", "schedule"}
        if unknown:
            raise ValueError(
                f"Unknown keys in schedule variant {index}: {sorted(unknown)}"
            )
        name = variant.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                f"Schedule variant {index} needs a filesystem-safe name"
            )
        if name in names:
            raise ValueError(f"Duplicate schedule variant name: {name}")
        schedule = variant.get("schedule")
        if schedule is not None and not isinstance(schedule, dict):
            raise TypeError(f"Schedule for variant {name} must be an object or null")
        names.append(name)
        normalized.append({"name": name, "schedule": schedule})
    if "baseline" not in names:
        raise ValueError("Schedule variants must include a fixed-scale 'baseline'")
    baseline = normalized[names.index("baseline")]
    if baseline["schedule"] is not None:
        raise ValueError("The 'baseline' variant must use schedule=null")
    return normalized


def load_occupancy(path: Path) -> torch.Tensor:
    positions = utils3d.io.read_ply(str(path))[0]
    coords = ((torch.as_tensor(positions) + 0.5) * RESOLUTION).int()
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.numel() == 0:
        raise ValueError(f"Invalid voxel coordinates in {path}: {tuple(coords.shape)}")
    if coords.min().item() < 0 or coords.max().item() >= RESOLUTION:
        raise ValueError(
            f"Voxel coordinates out of range in {path}: "
            f"{coords.min().item()}..{coords.max().item()}"
        )
    occupancy = torch.zeros(RESOLUTION, RESOLUTION, RESOLUTION, dtype=torch.bool)
    occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occupancy


def coords_to_occupancy(
    coords: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    xyz = coords[:, 1:].long()
    occupancy = torch.zeros_like(reference)
    occupancy[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return occupancy


def occupancy_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict:
    intersection = int((prediction & reference).sum().item())
    union = int((prediction | reference).sum().item())
    pred_count = int(prediction.sum().item())
    ref_count = int(reference.sum().item())
    precision = intersection / pred_count if pred_count else 0.0
    recall = intersection / ref_count if ref_count else 0.0
    return {
        "iou": intersection / union if union else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def _region_recall(prediction: torch.Tensor, region: torch.Tensor) -> Optional[float]:
    count = int(region.sum().item())
    if count == 0:
        return None
    return int((prediction & region).sum().item()) / count


def repair_metrics(
    prediction: torch.Tensor,
    mesh1: torch.Tensor,
    mesh2: torch.Tensor,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict:
    """Measure fill, removal, and preservation in task-specific voxel regions."""
    fill_region = mesh2 & ~mesh1
    remove_region = mesh1 & ~mesh2
    keep_region = mesh1 & mesh2
    fill_recall = _region_recall(prediction, fill_region)
    remove_success = _region_recall(~prediction, remove_region)
    keep_recall = _region_recall(prediction, keep_region)
    components = (fill_recall, remove_success, keep_recall)
    repair_score = (
        sum(weight * value for weight, value in zip(weights, components))
        if all(value is not None for value in components)
        else None
    )
    return {
        "fill_region_voxels": int(fill_region.sum().item()),
        "remove_region_voxels": int(remove_region.sum().item()),
        "keep_region_voxels": int(keep_region.sum().item()),
        "fill_recall": fill_recall,
        "remove_success": remove_success,
        "keep_recall": keep_recall,
        "repair_score": repair_score,
    }


def projection_strip(occupancy: torch.Tensor, label: str) -> Image.Image:
    array = occupancy.numpy()
    projections = [array.max(axis=axis) for axis in range(3)]
    strip = Image.new("RGB", (768, 286), "white")
    for axis, projection in enumerate(projections):
        view = Image.fromarray(
            projection.astype(np.uint8) * 255, mode="L"
        ).resize((256, 256), Image.Resampling.NEAREST).convert("RGB")
        panel = Image.new("RGB", (256, 286), "white")
        panel.paste(view, (0, 30))
        ImageDraw.Draw(panel).text((8, 8), f"{label} axis-{axis}", fill="black")
        strip.paste(panel, (axis * 256, 0))
    return strip


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(metadata_sources: Iterable[dict], keys: tuple[str, ...]) -> Any:
    for metadata in metadata_sources:
        for key in keys:
            if key in metadata and metadata[key] not in (None, ""):
                return metadata[key]
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, dict):
            for key in keys:
                if key in checkpoint and checkpoint[key] not in (None, ""):
                    return checkpoint[key]
    return None


def resolve_checkpoint_provenance(
    deploy_dir: Path,
    checkpoint_step_fallback: Optional[int],
    checkpoint_kind_fallback: Optional[str],
) -> dict:
    checkpoint_path = deploy_dir / "ckpts" / f"{CHECKPOINT_BASENAME}.safetensors"
    config_path = deploy_dir / "ckpts" / f"{CHECKPOINT_BASENAME}.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing deployed SS Flow checkpoint pair under {deploy_dir / 'ckpts'}"
        )

    sources = []
    with safe_open(checkpoint_path, framework="pt", device="cpu") as file:
        safetensors_metadata = file.metadata() or {}
    if safetensors_metadata:
        sources.append(safetensors_metadata)
    for candidate in (deploy_dir / "manifest.json", deploy_dir / "metadata.json"):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                sources.append(value)

    detected_step = _metadata_value(
        sources, ("checkpoint_step", "global_step", "step")
    )
    detected_kind = _metadata_value(
        sources, ("checkpoint_kind", "weight_kind", "weights")
    )
    if detected_step is not None:
        detected_step = int(detected_step)
    if detected_kind is not None:
        detected_kind = str(detected_kind).lower()
        if detected_kind not in {"raw", "ema"}:
            detected_kind = None

    if (
        detected_step is not None
        and checkpoint_step_fallback is not None
        and detected_step != checkpoint_step_fallback
    ):
        raise ValueError(
            f"Checkpoint metadata step {detected_step} conflicts with "
            f"--checkpoint_step {checkpoint_step_fallback}"
        )
    if (
        detected_kind is not None
        and checkpoint_kind_fallback is not None
        and detected_kind != checkpoint_kind_fallback
    ):
        raise ValueError(
            f"Checkpoint metadata kind {detected_kind} conflicts with "
            f"--checkpoint_kind {checkpoint_kind_fallback}"
        )

    checkpoint_step = (
        detected_step if detected_step is not None else checkpoint_step_fallback
    )
    checkpoint_kind = (
        detected_kind if detected_kind is not None else checkpoint_kind_fallback
    )
    return {
        "checkpoint_kind": checkpoint_kind or "unknown",
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "metadata_detected": bool(sources),
        "metadata_sources": sources,
    }


def build_ss_pipeline(
    deploy_dir: Path,
    sampler_params: dict,
) -> TrellisImageTo3DPipeline_ControlNet:
    ss_models = {
        "sparse_structure_flow_model": models.from_pretrained(
            str(deploy_dir / "ckpts" / CHECKPOINT_BASENAME)
        ),
        "sparse_structure_decoder": models.from_pretrained(
            str(deploy_dir / "ckpts/ss_dec_fine_tune_step2000_ControlNet")
        ),
    }
    pipeline = TrellisImageTo3DPipeline_ControlNet(
        models=ss_models,
        sparse_structure_sampler=samplers.FlowEulerGuidanceIntervalSampler(
            sigma_min=1e-5
        ),
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model="dinov2_vitl14_reg",
    )
    pipeline.sparse_structure_sampler_params = sampler_params
    pipeline.cuda()
    expected = {
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "image_cond_model",
    }
    if set(pipeline.models) != expected:
        raise RuntimeError(f"Unexpected models in SS-only pipeline: {pipeline.models.keys()}")
    return pipeline


def mean_present(rows: list[dict], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Evaluate at most this many metadata rows; 0 evaluates all rows.",
    )
    parser.add_argument("--base_control_scale", type=float, default=1.0)
    parser.add_argument("--variants_json", type=Path)
    parser.add_argument("--include_progress_variant", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.5, 1.0])
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--checkpoint_step", type=int)
    parser.add_argument("--checkpoint_kind", choices=["raw", "ema"])
    parser.add_argument(
        "--repair_weights",
        type=float,
        nargs=3,
        metavar=("FILL", "REMOVE", "KEEP"),
        default=[1 / 3, 1 / 3, 1 / 3],
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.base_control_scale < 0:
        raise ValueError("--base_control_scale must be non-negative")
    if any(weight < 0 for weight in args.repair_weights):
        raise ValueError("--repair_weights must be non-negative")
    weight_sum = sum(args.repair_weights)
    if weight_sum <= 0:
        raise ValueError("At least one repair weight must be positive")
    repair_weights = tuple(weight / weight_sum for weight in args.repair_weights)
    variants = load_schedule_variants(
        args.variants_json, include_progress=args.include_progress_variant
    )
    sampler_params = {
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "cfg_interval": args.cfg_interval,
        "rescale_t": args.rescale_t,
    }
    provenance = resolve_checkpoint_provenance(
        args.deploy_dir,
        checkpoint_step_fallback=args.checkpoint_step,
        checkpoint_kind_fallback=args.checkpoint_kind,
    )
    args.output_dir.mkdir(parents=True)

    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipeline = build_ss_pipeline(args.deploy_dir, sampler_params)
    flow_model = pipeline._validate_controlnet_flow_model()
    print(f"Loaded SS-only model keys: {sorted(pipeline.models)}", flush=True)
    print(f"Checkpoint provenance: {provenance}", flush=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv", dtype={"sha256": str})
    sample_ids = metadata["sha256"].tolist()
    if args.max_samples > 0:
        sample_ids = sample_ids[:args.max_samples]
    if not sample_ids:
        raise ValueError(f"No samples found in {args.data_dir / 'metadata.csv'}")
    all_rows = []
    schedule_traces = {}

    for sample_id in sample_ids:
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir()
        image_path = args.data_dir / "renders_cond" / sample_id / "up_normal.png"
        control_path = args.data_dir / "control_voxels" / f"{sample_id}.ply"
        target_path = args.data_dir / "target_voxels" / f"{sample_id}.ply"
        mesh1 = load_occupancy(control_path)
        mesh2 = load_occupancy(target_path)
        image = Image.open(image_path).convert("RGB")

        control = mesh1.float().unsqueeze(0).unsqueeze(0).to(
            device=pipeline.device,
            dtype=flow_model.control_encoder.input_layer.weight.dtype,
        )
        cond = pipeline.get_cond([image])
        with torch.no_grad():
            prepared_control = flow_model.prepare_control(control, batch_size=1)
        common_cpu = {
            "cond": cond["cond"].detach().cpu().contiguous(),
            "neg_cond": cond["neg_cond"].detach().cpu().contiguous(),
        }
        print(
            f"Sample {sample_id}: mesh1={int(mesh1.sum())}, "
            f"mesh2={int(mesh2.sum())}",
            flush=True,
        )

        predictions = {}
        sample_rows = []
        for variant in variants:
            variant_name = variant["name"]
            schedule = variant["schedule"]
            print(
                f"  variant={variant_name}, schedule={schedule}, seed={args.seed}",
                flush=True,
            )
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            coords = pipeline.sample_sparse_structure(
                cond,
                num_samples=1,
                prepared_control=prepared_control,
                control_scale=args.base_control_scale,
                control_schedule=schedule,
            )
            trace = pipeline.last_sparse_structure_control_trace
            if variant_name in schedule_traces:
                if trace != schedule_traces[variant_name]:
                    raise RuntimeError(
                        f"Control trace changed across samples for {variant_name}"
                    )
            else:
                schedule_traces[variant_name] = trace

            coords_cpu = coords.detach().cpu().to(torch.int32).contiguous()
            prediction = coords_to_occupancy(coords_cpu, mesh1)
            predictions[variant_name] = prediction
            variant_dir = sample_dir / variant_name
            variant_dir.mkdir()
            package_path = variant_dir / "ss_to_slat.safetensors"
            package_metadata = {
                "format": "trellis_controlnet_ss_to_slat_v2",
                "sample_id": sample_id,
                "checkpoint_kind": provenance["checkpoint_kind"],
                "checkpoint_step": (
                    str(provenance["checkpoint_step"])
                    if provenance["checkpoint_step"] is not None
                    else "unknown"
                ),
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "ss_seed": str(args.seed),
                "variant": variant_name,
                "base_control_scale": str(args.base_control_scale),
                "control_schedule": json.dumps(schedule),
            }
            save_file(
                {"coords": coords_cpu, **common_cpu},
                str(package_path),
                metadata=package_metadata,
            )
            xyz = coords_cpu[:, 1:].long()
            trimesh.points.PointCloud(
                xyz.float().numpy() / RESOLUTION - 0.5
            ).export(variant_dir / "ss_generated_sparse_structure.ply")

            target_stats = occupancy_metrics(prediction, mesh2)
            control_stats = occupancy_metrics(prediction, mesh1)
            repairs = repair_metrics(
                prediction, mesh1, mesh2, weights=repair_weights
            )
            row = {
                "sample_id": sample_id,
                "seed": args.seed,
                "variant": variant_name,
                "base_control_scale": args.base_control_scale,
                "control_schedule": json.dumps(schedule),
                "generated_voxels": int(prediction.sum().item()),
                "target_iou": target_stats["iou"],
                "target_precision": target_stats["precision"],
                "target_recall": target_stats["recall"],
                "target_f1": target_stats["f1"],
                "control_iou": control_stats["iou"],
                "control_precision": control_stats["precision"],
                "control_recall": control_stats["recall"],
                "control_f1": control_stats["f1"],
                **repairs,
                "package": str(package_path.relative_to(args.output_dir)),
                "package_sha256": file_sha256(package_path),
            }
            sample_rows.append(row)
            all_rows.append(row)

        baseline = predictions["baseline"]
        for row in sample_rows:
            prediction = predictions[row["variant"]]
            row["iou_vs_baseline"] = occupancy_metrics(
                prediction, baseline
            )["iou"]
            row["changed_voxels_vs_baseline"] = int(
                (prediction ^ baseline).sum().item()
            )
            variant_dir = sample_dir / row["variant"]
            manifest = {
                "format": "trellis_controlnet_ss_to_slat_v2",
                "provenance": provenance,
                "sample_id": sample_id,
                "seed": args.seed,
                "variant": row["variant"],
                "base_control_scale": args.base_control_scale,
                "control_schedule": json.loads(row["control_schedule"]),
                "control_schedule_trace": schedule_traces[row["variant"]],
                "ss_sampler_params": sampler_params,
                "repair_score_weights": {
                    "fill": repair_weights[0],
                    "remove": repair_weights[1],
                    "keep": repair_weights[2],
                },
                "image_preprocess": "FaceScan resize-only; no rembg",
                "metrics": row,
            }
            (variant_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        strips = [
            projection_strip(mesh1, "mesh1 control"),
            projection_strip(mesh2, "mesh2 target"),
            *[
                projection_strip(predictions[v["name"]], v["name"])
                for v in variants
            ],
        ]
        comparison = Image.new("RGB", (768, 286 * len(strips)), "white")
        for index, strip in enumerate(strips):
            comparison.paste(strip, (0, index * 286))
        comparison.save(sample_dir / "ss_schedule_comparison.png")
        shutil.copy2(image_path, sample_dir / "input_up_normal.png")
        shutil.copy2(control_path, sample_dir / "mesh1_control_voxels.ply")
        shutil.copy2(target_path, sample_dir / "mesh2_target_voxels.ply")

    write_csv(args.output_dir / "per_sample_metrics.csv", all_rows)
    aggregate = []
    for variant in variants:
        rows = [row for row in all_rows if row["variant"] == variant["name"]]
        aggregate.append({
            "variant": variant["name"],
            "samples": len(rows),
            "base_control_scale": args.base_control_scale,
            "control_schedule": json.dumps(variant["schedule"]),
            "mean_generated_voxels": mean_present(rows, "generated_voxels"),
            "mean_target_iou": mean_present(rows, "target_iou"),
            "mean_fill_recall": mean_present(rows, "fill_recall"),
            "mean_remove_success": mean_present(rows, "remove_success"),
            "mean_keep_recall": mean_present(rows, "keep_recall"),
            "mean_repair_score": mean_present(rows, "repair_score"),
            "mean_iou_vs_baseline": mean_present(rows, "iou_vs_baseline"),
            "mean_changed_voxels_vs_baseline": mean_present(
                rows, "changed_voxels_vs_baseline"
            ),
        })
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)

    summary = {
        "experiment": "SS ControlNet timestep-schedule repair evaluation",
        "provenance": provenance,
        "test_sample_ids": sample_ids,
        "seed": args.seed,
        "base_control_scale": args.base_control_scale,
        "schedule_variants": variants,
        "control_schedule_traces": schedule_traces,
        "ss_sampler_params": sampler_params,
        "repair_score_weights": {
            "fill": repair_weights[0],
            "remove": repair_weights[1],
            "keep": repair_weights[2],
        },
        "loaded_model_keys": sorted(pipeline.models),
        "slat_loaded": False,
        "slat_executed": False,
        "training_schedule_changed": False,
        "aggregate": aggregate,
        "results": all_rows,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
