"""Evaluate a raw SS ControlNet checkpoint at multiple control strengths.

This entry point deliberately loads and executes only DINOv2, SS Flow
ControlNet, and the matching SS Decoder. It exports each predicted sparse
structure together with the image-conditioning tensors required to continue
the SLat stage on another machine.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import pandas as pd
import torch
import trimesh
import utils3d
from PIL import Image, ImageDraw
from safetensors.torch import save_file

from trellis import models
from trellis.pipelines import samplers
from trellis.pipelines.trellis_image_to_3d_ControlNet import (
    TrellisImageTo3DPipeline_ControlNet,
)


RESOLUTION = 64
SAMPLER_PARAMS = {
    "steps": 25,
    "cfg_strength": 5.0,
    "cfg_interval": [0.5, 1.0],
    "rescale_t": 3.0,
}


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


def scale_tag(scale: float) -> str:
    return str(scale).replace(".", "p")


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


def build_ss_pipeline(deploy_dir: Path) -> TrellisImageTo3DPipeline_ControlNet:
    # Do not load SLat models: this experiment must end at decoded occupancy.
    ss_models = {
        "sparse_structure_flow_model": models.from_pretrained(
            str(deploy_dir / "ckpts/ss_flow_ControlNet")
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
    pipeline.sparse_structure_sampler_params = SAMPLER_PARAMS
    pipeline.cuda()
    expected = {
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "image_cond_model",
    }
    if set(pipeline.models) != expected:
        raise RuntimeError(f"Unexpected models in SS-only pipeline: {pipeline.models.keys()}")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--control_schedule_min_scale",
        type=float,
        default=None,
        help=(
            "Enable smoothstep timestep decay and keep this fraction of the "
            "base control_scale at late Flow timesteps. Omit for the fixed-scale "
            "baseline; use 0.1 for mild decay or 0.0 for strong decay."
        ),
    )
    parser.add_argument(
        "--control_schedule_full_strength_t",
        type=float,
        default=0.65,
        help="Flow timestep at and above which ControlNet remains at full strength.",
    )
    parser.add_argument(
        "--control_schedule_min_strength_t",
        type=float,
        default=0.25,
        help="Flow timestep at and below which ControlNet remains at its minimum strength.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipeline = build_ss_pipeline(args.deploy_dir)
    flow_model = pipeline._validate_controlnet_flow_model()
    control_schedule = None
    if args.control_schedule_min_scale is not None:
        control_schedule = {
            "name": "smoothstep",
            "full_strength_t": args.control_schedule_full_strength_t,
            "min_strength_t": args.control_schedule_min_strength_t,
            "min_scale": args.control_schedule_min_scale,
        }
    print(f"Loaded SS-only model keys: {sorted(pipeline.models)}", flush=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv", dtype={"sha256": str})
    sample_ids = metadata["sha256"].tolist()
    all_rows = []

    for sample_id in sample_ids:
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir()
        image_path = args.data_dir / "renders_cond" / sample_id / "up_normal.png"
        control_path = args.data_dir / "control_voxels" / f"{sample_id}.ply"
        target_path = args.data_dir / "target_voxels" / f"{sample_id}.ply"
        control_occ = load_occupancy(control_path)
        target_occ = load_occupancy(target_path)
        image = Image.open(image_path).convert("RGB")

        # The two unsqueeze operations create [B=1, C=1, R, R, R].
        control = control_occ.float().unsqueeze(0).unsqueeze(0).to(
            device=pipeline.device,
            dtype=flow_model.control_encoder.input_layer.weight.dtype,
        )
        # Match FaceScan training: resize-only image conditioning, no rembg.
        cond = pipeline.get_cond([image])
        with torch.no_grad():
            prepared_control = flow_model.prepare_control(control, batch_size=1)
        common_cpu = {
            "cond": cond["cond"].detach().cpu().contiguous(),
            "neg_cond": cond["neg_cond"].detach().cpu().contiguous(),
        }
        print(
            f"Sample {sample_id}: control={int(control_occ.sum())}, "
            f"target={int(target_occ.sum())}",
            flush=True,
        )

        # This extra baseline validates that scale zero exactly removes ControlNet.
        torch.manual_seed(args.seed)
        no_control_coords = pipeline.sample_sparse_structure(cond, num_samples=1)
        no_control_coords_cpu = no_control_coords.detach().cpu().to(torch.int32)
        no_control_xyz = no_control_coords_cpu[:, 1:].long()
        no_control_prediction = torch.zeros_like(control_occ)
        no_control_prediction[
            no_control_xyz[:, 0],
            no_control_xyz[:, 1],
            no_control_xyz[:, 2],
        ] = True
        predictions = {}
        sample_rows = []

        for scale in args.control_scales:
            print(
                f"  control_scale={scale}, schedule={control_schedule}, "
                f"seed={args.seed}",
                flush=True,
            )
            torch.manual_seed(args.seed)
            coords = pipeline.sample_sparse_structure(
                cond,
                num_samples=1,
                prepared_control=prepared_control,
                control_scale=scale,
                control_schedule=control_schedule,
            )
            coords_cpu = coords.detach().cpu().to(torch.int32).contiguous()
            if scale == 0.0 and not torch.equal(
                coords_cpu, no_control_coords_cpu
            ):
                raise RuntimeError(
                    f"{sample_id}: scale zero differs from the no-control baseline"
                )

            xyz = coords_cpu[:, 1:].long()
            prediction = torch.zeros_like(control_occ)
            prediction[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
            predictions[scale] = prediction
            scale_dir = sample_dir / f"scale_{scale_tag(scale)}"
            scale_dir.mkdir()
            package_path = scale_dir / "ss_to_slat.safetensors"
            save_file(
                {"coords": coords_cpu, **common_cpu},
                str(package_path),
                metadata={
                    "format": "trellis_controlnet_ss_to_slat_v1",
                    "sample_id": sample_id,
                    "checkpoint_kind": "raw",
                    "checkpoint_step": "4500",
                    "ss_seed": str(args.seed),
                    "control_scale": str(scale),
                    "control_schedule": json.dumps(control_schedule),
                },
            )
            points = xyz.float().numpy() / RESOLUTION - 0.5
            trimesh.points.PointCloud(points).export(
                scale_dir / "ss_generated_sparse_structure.ply"
            )
            target_stats = occupancy_metrics(prediction, target_occ)
            control_stats = occupancy_metrics(prediction, control_occ)
            row = {
                "sample_id": sample_id,
                "seed": args.seed,
                "control_scale": scale,
                "control_schedule": json.dumps(control_schedule),
                "generated_voxels": int(prediction.sum().item()),
                "target_iou": target_stats["iou"],
                "target_precision": target_stats["precision"],
                "target_recall": target_stats["recall"],
                "target_f1": target_stats["f1"],
                "control_iou": control_stats["iou"],
                "control_precision": control_stats["precision"],
                "control_recall": control_stats["recall"],
                "control_f1": control_stats["f1"],
                "package": str(package_path.relative_to(args.output_dir)),
                "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
            }
            sample_rows.append(row)
            all_rows.append(row)

        # Keep the historical metric names for CSV compatibility. The reference
        # is the explicit no-control prediction, which is exactly equivalent to
        # scale 0 and also works when --control_scales contains only 1.0.
        base = no_control_prediction
        for row in sample_rows:
            prediction = predictions[row["control_scale"]]
            row["iou_vs_scale_0"] = occupancy_metrics(prediction, base)["iou"]
            row["changed_voxels_vs_scale_0"] = int((prediction ^ base).sum().item())
            scale_dir = sample_dir / f"scale_{scale_tag(row['control_scale'])}"
            manifest = {
                "format": "trellis_controlnet_ss_to_slat_v1",
                "checkpoint_kind": "raw",
                "checkpoint_step": 4500,
                "sample_id": sample_id,
                "seed": args.seed,
                "control_scale": row["control_scale"],
                "control_schedule": control_schedule,
                "ss_sampler_params": SAMPLER_PARAMS,
                "image_preprocess": "FaceScan resize-only; no rembg",
                "metrics": row,
            }
            (scale_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        strips = [
            projection_strip(control_occ, "input control"),
            projection_strip(target_occ, "target"),
            *[
                projection_strip(predictions[scale], f"scale={scale}")
                for scale in args.control_scales
            ],
        ]
        comparison = Image.new("RGB", (768, 286 * len(strips)), "white")
        for index, strip in enumerate(strips):
            comparison.paste(strip, (0, index * 286))
        comparison.save(sample_dir / "ss_projection_comparison.png")
        shutil.copy2(image_path, sample_dir / "input_up_normal.png")
        shutil.copy2(control_path, sample_dir / "input_control_voxels.ply")
        shutil.copy2(target_path, sample_dir / "target_voxels.ply")

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate = []
    for scale in args.control_scales:
        rows = [row for row in all_rows if row["control_scale"] == scale]
        aggregate.append(
            {
                "control_scale": scale,
                "samples": len(rows),
                "mean_generated_voxels": sum(r["generated_voxels"] for r in rows)
                / len(rows),
                "mean_target_iou": sum(r["target_iou"] for r in rows) / len(rows),
                "mean_target_f1": sum(r["target_f1"] for r in rows) / len(rows),
                "mean_control_iou": sum(r["control_iou"] for r in rows) / len(rows),
                "mean_iou_vs_scale_0": sum(r["iou_vs_scale_0"] for r in rows)
                / len(rows),
                "mean_changed_voxels_vs_scale_0": sum(
                    r["changed_voxels_vs_scale_0"] for r in rows
                )
                / len(rows),
            }
        )
    with (args.output_dir / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    summary = {
        "experiment": (
            "SS-only fixed-noise ControlNet scale sweep on held-out test split"
            if control_schedule is None
            else "SS-only fixed-noise ControlNet timestep schedule on held-out test split"
        ),
        "checkpoint_kind": "raw",
        "checkpoint_step": 4500,
        "test_sample_ids": sample_ids,
        "seed": args.seed,
        "control_scales": args.control_scales,
        "control_schedule": control_schedule,
        "ss_sampler_params": SAMPLER_PARAMS,
        "loaded_model_keys": sorted(pipeline.models),
        "slat_loaded": False,
        "slat_executed": False,
        "scale_0_equals_no_control_for_all_samples": True,
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
