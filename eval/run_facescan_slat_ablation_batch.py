#!/usr/bin/env python3
"""Run split SLat inference for a mirrored FaceScan SS ablation tree.

This file only prepares SS-to-SLat handoff artifacts, invokes the existing
independent SLat Flow/decoder entrypoints, and mirrors their artifacts back to
``<output_root>/<sample_id>/<variant>/``. Model forward passes stay in the
existing inference scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import write_csv, write_json
from eval.common.ss_inference import save_image_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--flow-config",
        type=Path,
        default=Path("outputs/train/slat_flow_enc_finetune_dec_finetune/config.json"),
    )
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        default=Path(
            "outputs/train/slat_flow_enc_finetune_dec_finetune/ckpts/denoiser_step0015000.pt"
        ),
    )
    parser.add_argument(
        "--mesh-config",
        type=Path,
        default=Path("outputs/train/slat_dec_mesh_fine_tune/config.json"),
    )
    parser.add_argument(
        "--mesh-ckpt",
        type=Path,
        default=Path(
            "outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt"
        ),
    )
    parser.add_argument(
        "--gaussian-config",
        type=Path,
        default=Path(
            "outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8/config.json"
        ),
    )
    parser.add_argument(
        "--gaussian-ckpt",
        type=Path,
        default=Path(
            "outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8/ckpts/decoder_step0004000.pt"
        ),
    )
    parser.add_argument(
        "--stable3dgen-root", type=Path, default=Path("/root/autodl-tmp/Stable3DGen")
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--cfg-interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {row["sample_id"]: row for row in csv.DictReader(file)}


def scan_packages(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for package_path in sorted(source_root.glob("*/*/ss_to_slat.safetensors")):
        variant_dir = package_path.parent
        sample_dir = variant_dir.parent
        sample_id = sample_dir.name
        variant = variant_dir.name
        manifest_path = variant_dir / "manifest.json"
        sparse_path = variant_dir / "ss_generated_sparse_structure.ply"
        image_path = sample_dir / "input_up_normal.png"
        for path in (manifest_path, sparse_path, image_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        records.append(
            {
                "sample_id": sample_id,
                "variant": variant,
                "batch_id": f"{sample_id}__{variant}",
                "package_path": package_path.resolve(),
                "manifest_path": manifest_path.resolve(),
                "sparse_path": sparse_path.resolve(),
                "image_path": image_path.resolve(),
            }
        )
    if not records:
        raise ValueError(f"No SS-to-SLat packages under {source_root}")
    batch_ids = [record["batch_id"] for record in records]
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Duplicate sample/variant pairs in source tree")
    return records


def prepare_handoffs(
    records: list[dict[str, Any]], output_root: Path, batch_manifest: Path, default_seed: int
) -> None:
    manifest_rows: list[dict[str, Any]] = []
    condition_cache: dict[str, tuple[torch.Tensor, torch.Tensor, Path]] = {}
    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        variant = record["variant"]
        package_path = record["package_path"]
        source_manifest = json.loads(record["manifest_path"].read_text(encoding="utf-8"))
        package_hash = sha256(package_path)
        expected_hash = source_manifest.get("metrics", {}).get("package_sha256")
        if expected_hash and package_hash != expected_hash:
            raise ValueError(
                f"{sample_id}/{variant}: package checksum mismatch: "
                f"expected={expected_hash}, actual={package_hash}"
            )
        tensors = load_file(package_path, device="cpu")
        missing = {"coords", "cond", "neg_cond"}.difference(tensors)
        if missing:
            raise KeyError(f"{sample_id}/{variant}: missing tensors {sorted(missing)}")
        coords4 = tensors["coords"].detach().cpu().numpy().astype(np.int32, copy=False)
        if coords4.ndim != 2 or coords4.shape[1] != 4:
            raise ValueError(f"{sample_id}/{variant}: invalid coords shape {coords4.shape}")
        if not np.all(coords4[:, 0] == 0):
            raise ValueError(f"{sample_id}/{variant}: nonzero batch indices")
        coords = coords4[:, 1:]
        if (
            coords.shape[0] == 0
            or coords.min() < 0
            or coords.max() >= 64
            or np.unique(coords, axis=0).shape[0] != coords.shape[0]
        ):
            raise ValueError(f"{sample_id}/{variant}: invalid resolution-64 coordinates")

        cond = tensors["cond"]
        neg_cond = tensors["neg_cond"]
        if sample_id in condition_cache:
            cached_cond, cached_neg, cached_path = condition_cache[sample_id]
            if not torch.equal(cond, cached_cond) or not torch.equal(neg_cond, cached_neg):
                raise ValueError(f"{sample_id}/{variant}: image condition differs within one ID")
        else:
            cached_path = output_root / ".shared_conditions" / f"{sample_id}.npz"
            save_image_condition(cached_path, {"cond": cond, "neg_cond": neg_cond})
            condition_cache[sample_id] = (cond.clone(), neg_cond.clone(), cached_path)

        branch = output_root / sample_id / variant
        handoff = branch / "input_handoff"
        coords_path = handoff / "coords.npz"
        condition_path = handoff / "condition_features.npz"
        image_path = handoff / "input_up_normal.png"
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(coords_path, coords=coords)
        link_or_copy(cached_path, condition_path)
        link_or_copy(record["image_path"], image_path)
        link_or_copy(package_path, handoff / "ss_to_slat.safetensors")
        link_or_copy(record["manifest_path"], handoff / "source_manifest.json")
        link_or_copy(record["sparse_path"], handoff / "ss_generated_sparse_structure.ply")

        seed = int(source_manifest.get("seed", default_seed))
        handoff_summary = {
            "sample_id": sample_id,
            "variant": variant,
            "batch_id": record["batch_id"],
            "num_coords": int(coords.shape[0]),
            "coord_min": coords.min(axis=0).astype(int).tolist(),
            "coord_max": coords.max(axis=0).astype(int).tolist(),
            "seed": seed,
            "package_sha256": package_hash,
            "mask": source_manifest.get("mask"),
            "condition_shape": list(cond.shape),
        }
        write_json(handoff / "handoff_manifest.json", handoff_summary)
        record.update(
            {
                "coords_path": coords_path.resolve(),
                "condition_path": condition_path.resolve(),
                "prepared_image_path": image_path.resolve(),
                "seed": seed,
                "num_coords": int(coords.shape[0]),
                "package_sha256": package_hash,
                "source_manifest": source_manifest,
            }
        )
        manifest_rows.append(
            {
                "stage": "ss_controlnet_package",
                "source_stage": "ss_flow",
                "sample_id": record["batch_id"],
                "dataset_index": index,
                "coords_path": str(coords_path.resolve()),
                "condition_image_path": str(image_path.resolve()),
                "prepared_condition_path": str(image_path.resolve()),
                "condition_preprocessed": True,
                "condition_features_path": str(condition_path.resolve()),
                "rng_state_path": "",
                "seed": seed,
                "failed": False,
                "error": "",
            }
        )
        print(f"[prepare {index + 1}/{len(records)}] {sample_id}/{variant}", flush=True)
    write_csv(batch_manifest, manifest_rows)


def run_command(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_stages(args: argparse.Namespace, batch_manifest: Path, work: Path) -> None:
    common = ["--device", args.device, "--fail_on_error"]
    if args.resume:
        common.append("--skip_existing")
    run_command(
        [
            args.python,
            "-m",
            "eval.slat_flow_inference",
            "--config",
            str(absolute(args.flow_config)),
            "--flow_ckpt",
            str(absolute(args.flow_ckpt)),
            "--ss_manifest",
            str(batch_manifest),
            "--output_dir",
            str(work / "flow"),
            "--stable3dgen_root",
            str(absolute(args.stable3dgen_root)),
            "--num_samples",
            "0",
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--cfg_strength",
            str(args.cfg_strength),
            "--cfg_interval",
            str(args.cfg_interval[0]),
            str(args.cfg_interval[1]),
            "--rescale_t",
            str(args.rescale_t),
            *common,
        ]
    )
    for name, config, checkpoint, output_format in (
        ("mesh_decoder", args.mesh_config, args.mesh_ckpt, "mesh"),
        ("gaussian_decoder", args.gaussian_config, args.gaussian_ckpt, "gaussian"),
    ):
        run_command(
            [
                args.python,
                "-m",
                "eval.slat_decoder_inference",
                "--config",
                str(absolute(config)),
                "--decoder_ckpt",
                str(absolute(checkpoint)),
                "--latent_manifest",
                str(work / "flow" / "manifest.csv"),
                "--output_dir",
                str(work / name),
                "--format",
                output_format,
                *common,
            ]
        )


def materialize_and_validate(
    records: list[dict[str, Any]], output_root: Path, work: Path, args: argparse.Namespace
) -> dict[str, Any]:
    flow_rows = read_rows(work / "flow" / "manifest.csv")
    mesh_rows = read_rows(work / "mesh_decoder" / "manifest.csv")
    gaussian_rows = read_rows(work / "gaussian_decoder" / "manifest.csv")
    aggregate_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        batch_id = record["batch_id"]
        sample_id = record["sample_id"]
        variant = record["variant"]
        branch = output_root / sample_id / variant
        flow_row = flow_rows[batch_id]
        mesh_row = mesh_rows[batch_id]
        gaussian_row = gaussian_rows[batch_id]
        stage_failed = any(
            str(row.get("failed", "")).strip().lower() in {"1", "true", "yes"}
            for row in (flow_row, mesh_row, gaussian_row)
        )
        if stage_failed:
            error = {
                "sample_id": sample_id,
                "variant": variant,
                "flow_error": flow_row.get("error", ""),
                "mesh_error": mesh_row.get("error", ""),
                "gaussian_error": gaussian_row.get("error", ""),
            }
            failed.append(error)
            continue

        flow_dir = branch / "flow"
        mesh_dir = branch / "mesh_decoder"
        gaussian_dir = branch / "gaussian_decoder"
        flow_stats_source = work / "flow" / "samples" / batch_id / "stats.json"
        link_or_copy(Path(flow_row["latent_path"]), flow_dir / "latent.npz")
        link_or_copy(Path(flow_row["prepared_condition_path"]), flow_dir / "cond.png")
        link_or_copy(flow_stats_source, flow_dir / "stats.json")
        link_or_copy(Path(mesh_row["artifact_path"]), mesh_dir / "mesh.ply")
        link_or_copy(Path(mesh_row["stats_path"]), mesh_dir / "stats.json")
        link_or_copy(Path(gaussian_row["artifact_path"]), gaussian_dir / "gaussian.ply")
        link_or_copy(Path(gaussian_row["stats_path"]), gaussian_dir / "stats.json")

        source_tensors = load_file(record["package_path"], device="cpu")
        source_coords = source_tensors["coords"][:, 1:].numpy().astype(np.int32, copy=False)
        with np.load(record["coords_path"], allow_pickle=False) as data:
            handoff_coords = data["coords"]
        with np.load(flow_dir / "latent.npz", allow_pickle=False) as data:
            flow_coords = data["coords"]
            flow_shape = list(data["feats"].shape)
        with np.load(record["condition_path"], allow_pickle=False) as data:
            condition_matches = np.array_equal(
                data["cond"], source_tensors["cond"].numpy()
            ) and np.array_equal(data["neg_cond"], source_tensors["neg_cond"].numpy())
        validation = {
            "sample_id": sample_id,
            "variant": variant,
            "num_coords": int(source_coords.shape[0]),
            "handoff_matches_source_coords": bool(np.array_equal(handoff_coords, source_coords)),
            "flow_matches_source_coords": bool(np.array_equal(flow_coords, source_coords)),
            "condition_features_match_source": bool(condition_matches),
            "condition_image_matches_source": sha256(record["prepared_image_path"])
            == sha256(record["image_path"]),
            "latent_feats_shape": flow_shape,
            "mesh_vertices": int(mesh_row["num_vertices"]),
            "mesh_faces": int(mesh_row["num_faces"]),
            "num_gaussians": int(gaussian_row["num_gaussians"]),
        }
        validation["passed"] = all(
            validation[key]
            for key in (
                "handoff_matches_source_coords",
                "flow_matches_source_coords",
                "condition_features_match_source",
                "condition_image_matches_source",
            )
        )
        if not validation["passed"]:
            failed.append(validation)
        write_json(branch / "validation.json", validation)
        validations.append(validation)
        row = {
            "sample_id": sample_id,
            "variant": variant,
            "batch_id": batch_id,
            "num_coords": int(source_coords.shape[0]),
            "latent_path": str((flow_dir / "latent.npz").resolve()),
            "mesh_path": str((mesh_dir / "mesh.ply").resolve()),
            "gaussian_path": str((gaussian_dir / "gaussian.ply").resolve()),
            "mesh_vertices": int(mesh_row["num_vertices"]),
            "mesh_faces": int(mesh_row["num_faces"]),
            "num_gaussians": int(gaussian_row["num_gaussians"]),
            "seed": record["seed"],
            "failed": not validation["passed"],
            "error": "" if validation["passed"] else "validation failed",
        }
        aggregate_rows.append(row)
        write_json(branch / "result_manifest.json", row)
        print(f"[collect {index + 1}/{len(records)}] {sample_id}/{variant}", flush=True)

    summary = {
        "status": "completed" if not failed else "completed_with_failures",
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "num_records": len(records),
        "successful_count": len(records) - len(failed),
        "failed_count": len(failed),
        "sampler": {
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
            "cfg_interval": list(args.cfg_interval),
            "rescale_t": args.rescale_t,
            "seed": args.seed,
        },
        "checkpoints": {
            "flow": str(absolute(args.flow_ckpt)),
            "mesh_decoder": str(absolute(args.mesh_ckpt)),
            "gaussian_decoder": str(absolute(args.gaussian_ckpt)),
        },
    }
    write_csv(output_root / "manifest.csv", aggregate_rows)
    write_json(output_root / "failed_samples.json", failed)
    write_json(output_root / "validation_summary.json", validations)
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.source_root = absolute(args.source_root)
    args.output_root = absolute(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = args.output_root / ".batch_work"
    work.mkdir(parents=True, exist_ok=True)
    batch_manifest = work / "ss_manifest.csv"
    records = scan_packages(args.source_root)
    prepare_handoffs(records, args.output_root, batch_manifest, args.seed)
    write_json(
        args.output_root / "run_config.json",
        {
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "num_records": len(records),
            "sample_variants": [
                {"sample_id": row["sample_id"], "variant": row["variant"]}
                for row in records
            ],
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    run_stages(args, batch_manifest, work)
    summary = materialize_and_validate(records, args.output_root, work, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
