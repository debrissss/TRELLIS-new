"""SS-only layer-injection ablation for FaceScan ControlNet.

All active residual injections use one shared base strength. The experiment
compares grouped masks, leave-one-out masks, and one-hot masks without loading
or executing any SLat model.
"""

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image
from safetensors.torch import save_file
from scipy import ndimage

from fine_tuning.eval_face_scan_ControlNet_ss_scale_sweep import (
    RESOLUTION,
    SAMPLER_PARAMS,
    build_ss_pipeline,
    load_occupancy,
    occupancy_metrics,
    projection_strip,
)


def build_masks(active_scale: float) -> list[dict]:
    on, off = active_scale, 0.0
    masks = [
        {"name": "none", "family": "grouped", "scales": [off] * 8},
        {"name": "full_0p75", "family": "grouped", "scales": [on] * 8},
        {"name": "front4", "family": "grouped", "scales": [on] * 4 + [off] * 4},
        {"name": "back4", "family": "grouped", "scales": [off] * 4 + [on] * 4},
        {
            "name": "middle4",
            "family": "grouped",
            "scales": [off, off, on, on, on, on, off, off],
        },
        {
            "name": "even",
            "family": "grouped",
            "scales": [on, off, on, off, on, off, on, off],
        },
        {
            "name": "odd",
            "family": "grouped",
            "scales": [off, on, off, on, off, on, off, on],
        },
        {"name": "front2", "family": "grouped", "scales": [on, on] + [off] * 6},
        {"name": "back2", "family": "grouped", "scales": [off] * 6 + [on, on]},
    ]
    for index in range(8):
        scales = [on] * 8
        scales[index] = off
        masks.append(
            {
                "name": f"drop_{index}",
                "family": "drop",
                "layer": index,
                "scales": scales,
            }
        )
    for index in range(8):
        scales = [off] * 8
        scales[index] = on
        masks.append(
            {
                "name": f"only_{index}",
                "family": "only",
                "layer": index,
                "scales": scales,
            }
        )
    names = [mask["name"] for mask in masks]
    if len(names) != len(set(names)):
        raise RuntimeError("Layer-ablation mask names must be unique")
    return masks


def component_stats(mask: torch.Tensor) -> dict:
    array = mask.numpy()
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(array, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max())
    else:
        largest = 0
    total = int(array.sum())
    return {
        "components": int(count),
        "largest_component_voxels": largest,
        "largest_component_ratio": largest / total if total else 0.0,
    }


def topology_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    pred_components = component_stats(prediction)
    false_negative = target & ~prediction
    fn_components = component_stats(false_negative)

    # The voxel data represents a surface. Filling background regions not
    # connected to the volume boundary is therefore only a closure proxy, not
    # a literal semantic count of facial holes.
    pred_np = prediction.numpy()
    filled = ndimage.binary_fill_holes(
        pred_np, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    enclosed = torch.from_numpy(filled & ~pred_np)
    enclosed_components = component_stats(enclosed)
    return {
        "prediction_components": pred_components["components"],
        "prediction_largest_component_ratio": pred_components[
            "largest_component_ratio"
        ],
        "false_negative_voxels": int(false_negative.sum().item()),
        "false_negative_components": fn_components["components"],
        "false_negative_largest_component_voxels": fn_components[
            "largest_component_voxels"
        ],
        "enclosed_volume_proxy_voxels": int(enclosed.sum().item()),
        "enclosed_volume_proxy_components": enclosed_components["components"],
    }


def save_comparison(
    path: Path,
    names: list[str],
    predictions: dict[str, torch.Tensor],
    control: torch.Tensor,
    target: torch.Tensor,
) -> None:
    strips = [projection_strip(control, "input control"), projection_strip(target, "target")]
    strips.extend(projection_strip(predictions[name], name) for name in names)
    canvas = Image.new("RGB", (768, 286 * len(strips)), "white")
    for index, strip in enumerate(strips):
        canvas.paste(strip, (0, index * 286))
    canvas.save(path)


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active_scale", type=float, default=0.75)
    args = parser.parse_args()
    if args.active_scale != 0.75:
        raise ValueError("This experiment is defined against full_0p75")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    masks = build_masks(args.active_scale)
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipeline = build_ss_pipeline(args.deploy_dir)
    flow_model = pipeline._validate_controlnet_flow_model()
    metadata = pd.read_csv(args.data_dir / "metadata.csv", dtype={"sha256": str})
    sample_ids = metadata["sha256"].tolist()
    if len(sample_ids) != 2:
        raise ValueError(f"Expected exactly two held-out samples, got {len(sample_ids)}")
    print(f"Loaded SS-only models: {sorted(pipeline.models)}", flush=True)
    print(f"Masks: {len(masks)}; samples: {sample_ids}", flush=True)

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
        control = control_occ.float().unsqueeze(0).unsqueeze(0).to(
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

        # Verify the all-zero mask exactly matches the no-ControlNet path.
        torch.manual_seed(args.seed)
        no_control_coords = pipeline.sample_sparse_structure(cond, num_samples=1)
        predictions = {}
        coords_by_mask = {}
        sample_rows = []
        print(f"\nSample {sample_id}", flush=True)
        for mask in masks:
            print(f"  {mask['name']}: {mask['scales']}", flush=True)
            torch.manual_seed(args.seed)
            coords = pipeline.sample_sparse_structure(
                cond,
                num_samples=1,
                prepared_control=prepared_control,
                control_scale=mask["scales"],
            )
            coords_cpu = coords.detach().cpu().to(torch.int32).contiguous()
            if mask["name"] == "none" and not torch.equal(
                coords_cpu, no_control_coords.detach().cpu().to(torch.int32)
            ):
                raise RuntimeError(f"{sample_id}: zero mask differs from control=None")
            xyz = coords_cpu[:, 1:].long()
            prediction = torch.zeros_like(control_occ)
            prediction[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
            predictions[mask["name"]] = prediction
            coords_by_mask[mask["name"]] = coords_cpu

            mask_dir = sample_dir / mask["name"]
            mask_dir.mkdir()
            package_path = mask_dir / "ss_to_slat.safetensors"
            save_file(
                {"coords": coords_cpu, **common_cpu},
                str(package_path),
                metadata={
                    "format": "trellis_controlnet_ss_to_slat_v1",
                    "sample_id": sample_id,
                    "checkpoint_kind": "raw",
                    "checkpoint_step": "4500",
                    "ss_seed": str(args.seed),
                    "layer_mask_name": mask["name"],
                    "control_scale": json.dumps(mask["scales"]),
                },
            )
            points = xyz.float().numpy() / RESOLUTION - 0.5
            trimesh.points.PointCloud(points).export(
                mask_dir / "ss_generated_sparse_structure.ply"
            )
            target_stats = occupancy_metrics(prediction, target_occ)
            control_stats = occupancy_metrics(prediction, control_occ)
            row = {
                "sample_id": sample_id,
                "seed": args.seed,
                "mask_name": mask["name"],
                "mask_family": mask["family"],
                "layer": mask.get("layer", ""),
                "control_scales": json.dumps(mask["scales"]),
                "active_layers": sum(value != 0 for value in mask["scales"]),
                "generated_voxels": int(prediction.sum().item()),
                "target_iou": target_stats["iou"],
                "target_precision": target_stats["precision"],
                "target_recall": target_stats["recall"],
                "target_f1": target_stats["f1"],
                "control_iou": control_stats["iou"],
                **topology_metrics(prediction, target_occ),
                "package": str(package_path.relative_to(args.output_dir)),
                "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
            }
            sample_rows.append(row)
            all_rows.append(row)

        full = predictions["full_0p75"]
        full_row = next(row for row in sample_rows if row["mask_name"] == "full_0p75")
        for row in sample_rows:
            prediction = predictions[row["mask_name"]]
            row["iou_vs_full_0p75"] = occupancy_metrics(prediction, full)["iou"]
            row["changed_voxels_vs_full_0p75"] = int((prediction ^ full).sum().item())
            row["added_voxels_vs_full_0p75"] = int((prediction & ~full).sum().item())
            row["removed_voxels_vs_full_0p75"] = int((full & ~prediction).sum().item())
            row["delta_target_iou_vs_full_0p75"] = row["target_iou"] - full_row[
                "target_iou"
            ]
            row["delta_target_f1_vs_full_0p75"] = row["target_f1"] - full_row[
                "target_f1"
            ]
            row["delta_false_negative_voxels_vs_full_0p75"] = (
                row["false_negative_voxels"] - full_row["false_negative_voxels"]
            )
            row["delta_enclosed_volume_proxy_vs_full_0p75"] = (
                row["enclosed_volume_proxy_voxels"]
                - full_row["enclosed_volume_proxy_voxels"]
            )
            mask = next(item for item in masks if item["name"] == row["mask_name"])
            manifest = {
                "format": "trellis_controlnet_ss_to_slat_v1",
                "checkpoint_kind": "raw",
                "checkpoint_step": 4500,
                "sample_id": sample_id,
                "seed": args.seed,
                "active_scale": args.active_scale,
                "mask": mask,
                "ss_sampler_params": SAMPLER_PARAMS,
                "slat_loaded": False,
                "slat_executed": False,
                "metrics": row,
            }
            (sample_dir / row["mask_name"] / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        grouped = [mask["name"] for mask in masks if mask["family"] == "grouped"]
        drops = ["full_0p75"] + [
            mask["name"] for mask in masks if mask["family"] == "drop"
        ]
        only = ["none"] + [
            mask["name"] for mask in masks if mask["family"] == "only"
        ]
        save_comparison(
            sample_dir / "comparison_grouped.png",
            grouped,
            predictions,
            control_occ,
            target_occ,
        )
        save_comparison(
            sample_dir / "comparison_drop.png",
            drops,
            predictions,
            control_occ,
            target_occ,
        )
        save_comparison(
            sample_dir / "comparison_only.png",
            only,
            predictions,
            control_occ,
            target_occ,
        )
        shutil.copy2(image_path, sample_dir / "input_up_normal.png")
        shutil.copy2(control_path, sample_dir / "input_control_voxels.ply")
        shutil.copy2(target_path, sample_dir / "target_voxels.ply")

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate = []
    for mask in masks:
        rows = [row for row in all_rows if row["mask_name"] == mask["name"]]
        aggregate.append(
            {
                "mask_name": mask["name"],
                "mask_family": mask["family"],
                "layer": mask.get("layer", ""),
                "control_scales": json.dumps(mask["scales"]),
                "active_layers": sum(value != 0 for value in mask["scales"]),
                "mean_generated_voxels": mean(rows, "generated_voxels"),
                "mean_target_iou": mean(rows, "target_iou"),
                "mean_target_f1": mean(rows, "target_f1"),
                "mean_control_iou": mean(rows, "control_iou"),
                "mean_false_negative_voxels": mean(rows, "false_negative_voxels"),
                "mean_false_negative_largest_component_voxels": mean(
                    rows, "false_negative_largest_component_voxels"
                ),
                "mean_prediction_components": mean(rows, "prediction_components"),
                "mean_prediction_largest_component_ratio": mean(
                    rows, "prediction_largest_component_ratio"
                ),
                "mean_enclosed_volume_proxy_voxels": mean(
                    rows, "enclosed_volume_proxy_voxels"
                ),
                "mean_iou_vs_full_0p75": mean(rows, "iou_vs_full_0p75"),
                "mean_changed_voxels_vs_full_0p75": mean(
                    rows, "changed_voxels_vs_full_0p75"
                ),
                "mean_delta_target_iou_vs_full_0p75": mean(
                    rows, "delta_target_iou_vs_full_0p75"
                ),
                "mean_delta_target_f1_vs_full_0p75": mean(
                    rows, "delta_target_f1_vs_full_0p75"
                ),
                "mean_delta_false_negative_voxels_vs_full_0p75": mean(
                    rows, "delta_false_negative_voxels_vs_full_0p75"
                ),
            }
        )
    with (args.output_dir / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    layer_effects = [row for row in aggregate if row["mask_family"] in {"drop", "only"}]
    with (args.output_dir / "layer_effects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(layer_effects[0]))
        writer.writeheader()
        writer.writerows(layer_effects)

    summary = {
        "experiment": "SS-only ControlNet layer injection ablation",
        "checkpoint_kind": "raw",
        "checkpoint_step": 4500,
        "baseline": "full_0p75",
        "active_scale": args.active_scale,
        "test_sample_ids": sample_ids,
        "seed": args.seed,
        "masks": masks,
        "ss_sampler_params": SAMPLER_PARAMS,
        "loaded_model_keys": sorted(pipeline.models),
        "slat_loaded": False,
        "slat_executed": False,
        "zero_mask_equals_no_control_for_all_samples": True,
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
