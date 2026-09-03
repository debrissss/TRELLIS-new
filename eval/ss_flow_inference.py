#!/usr/bin/env python3
"""Run image-conditioned sparse-structure flow as an independent stage."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json
from eval.common.ss_inference import (
    latent_stats,
    load_configured_model,
    require_device,
    require_nonempty_unique_samples,
    resolve_input_records,
    sample_output_dir,
    save_image_condition,
    save_ss_latent,
    save_torch_cpu_rng_state,
    write_input_manifest,
    write_stage_result,
)


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SS flow training config JSON")
    parser.add_argument(
        "--flow_ckpt",
        required=True,
        help="Denoiser .pt checkpoint or from_pretrained-compatible model prefix",
    )
    parser.add_argument(
        "--stable3dgen_root",
        type=Path,
        default=DEFAULT_STABLE3DGEN_ROOT,
        help="Current Stable3DGen checkout providing Hi3DGen preprocessing and sampler",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--data_dir",
        type=Path,
        help="Dataset root containing metadata.csv and renders_cond/",
    )
    inputs.add_argument("--input_manifest", type=Path, help="Fixed selected_samples.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=16, help="<=0 selects all valid samples")
    parser.add_argument("--indices", default=None, help="Comma-separated indices in the valid-sample list")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument(
        "--no_preprocess_image",
        dest="preprocess_image",
        action="store_false",
        help="Pass the selected condition image directly to DINO preprocessing",
    )
    parser.set_defaults(preprocess_image=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def _load_hi3dgen_components(stable3dgen_root: Path):
    stable3dgen_root = stable3dgen_root.expanduser().resolve()
    package_dir = stable3dgen_root / "hi3dgen"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Stable3DGen hi3dgen package not found: {package_dir}")
    root_text = str(stable3dgen_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from hi3dgen.pipelines import Hi3DGenPipeline
    from hi3dgen.pipelines import samplers

    return Hi3DGenPipeline, samplers


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preprocess_with_stable3dgen(
    pipeline: Any,
    image: Any,
    *,
    resolution: int,
    stable3dgen_root: Path,
):
    """Call Stable3DGen preprocessing with its relative weight paths intact."""

    previous_cwd = Path.cwd()
    try:
        os.chdir(stable3dgen_root)
        return pipeline.preprocess_image(image, resolution=resolution)
    finally:
        os.chdir(previous_cwd)


def _build_flow_pipeline(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: Any,
):
    Hi3DGenPipeline, stable_samplers = _load_hi3dgen_components(args.stable3dgen_root)
    flow_model = load_configured_model(
        config,
        model_key="denoiser",
        checkpoint=args.flow_ckpt,
        device=device,
    )
    trainer_args = config.get("trainer", {}).get("args", {})
    sigma_min = float(trainer_args.get("sigma_min", 1e-5))
    image_cond_model = trainer_args.get("image_cond_model", "dinov2_vitl14_reg")
    sampler = stable_samplers.FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    # Minimal pipeline: only SS Flow and DINO are registered. No SS decoder or
    # SLat component is constructed or loaded in this stage.
    pipeline = Hi3DGenPipeline(
        models={"sparse_structure_flow_model": flow_model},
        sparse_structure_sampler=sampler,
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model=image_cond_model,
    )
    pipeline.sparse_structure_sampler_params = {}
    pipeline.to(device)
    return pipeline, sigma_min, image_cond_model


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    config = load_json(args.config)
    records = resolve_input_records(
        data_dir=args.data_dir,
        input_manifest=args.input_manifest,
        num_samples=args.num_samples,
        seed=args.seed,
        indices=args.indices,
        require_voxel=False,
        require_condition=True,
    )
    require_nonempty_unique_samples(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_input_manifest(args.output_dir / "selected_samples.csv", records)

    device = require_device(args.device)
    pipeline, sigma_min, image_cond_model = _build_flow_pipeline(args, config, device)
    flow_model = pipeline.models["sparse_structure_flow_model"]
    sampler_params = {
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "cfg_interval": tuple(args.cfg_interval),
        "rescale_t": args.rescale_t,
        "verbose": args.verbose,
    }

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_index, record in enumerate(records):
            sample_id = record["sample_id"]
            condition_path = Path(record["condition_image_path"])
            sample_seed = args.seed + sample_index
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            latent_path = sample_dir / "latent.npz"
            prepared_path = sample_dir / "cond.png"
            condition_features_path = sample_dir / "condition_features.npz"
            rng_state_path = sample_dir / "rng_state.npz"
            stats_path = sample_dir / "stats.json"
            base_row = {
                "stage": "ss_flow",
                "sample_id": sample_id,
                "sample_index": sample_index,
                "dataset_index": record["dataset_index"],
                "voxel_path": record.get("voxel_path", ""),
                "condition_image_path": str(condition_path),
                "prepared_condition_path": str(prepared_path.resolve()),
                "condition_preprocessed": bool(args.preprocess_image),
                "condition_features_path": str(condition_features_path.resolve()),
                "latent_path": str(latent_path.resolve()),
                "latent_key": "z_s",
                "rng_state_path": str(rng_state_path.resolve()),
                "seed": sample_seed,
                "checkpoint": args.flow_ckpt,
            }
            if (
                args.skip_existing
                and latent_path.is_file()
                and prepared_path.is_file()
                and condition_features_path.is_file()
                and rng_state_path.is_file()
                and stats_path.is_file()
            ):
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                with Image.open(condition_path) as source_image:
                    image = source_image.convert("RGBA")
                if args.preprocess_image:
                    prepared = _preprocess_with_stable3dgen(
                        pipeline,
                        image,
                        resolution=args.preprocess_resolution,
                        stable3dgen_root=args.stable3dgen_root.expanduser().resolve(),
                    )
                else:
                    prepared = image

                cond = pipeline.get_cond([prepared])
                save_image_condition(condition_features_path, cond)
                resolution = int(flow_model.resolution)
                # Stable3DGen seeds after image conditioning and creates the
                # initial noise on CPU before moving it to the pipeline device.
                # Preserve both details so a split run consumes the exact same
                # default-generator sequence as Hi3DGenPipeline.run().
                torch.manual_seed(sample_seed)
                noise = torch.randn(
                    1,
                    int(flow_model.in_channels),
                    resolution,
                    resolution,
                    resolution,
                ).to(device)
                save_torch_cpu_rng_state(rng_state_path, torch.get_rng_state())
                result = pipeline.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    **cond,
                    **sampler_params,
                )
                z_s = result["samples"]
                save_ss_latent(latent_path, z_s[0])
                sample_dir.mkdir(parents=True, exist_ok=True)
                prepared.save(prepared_path)
                stats = {
                    "stage": "ss_flow",
                    "sample_id": sample_id,
                    "condition_image_path": str(condition_path),
                    "prepared_condition_path": str(prepared_path.resolve()),
                    "condition_features_path": str(
                        condition_features_path.resolve()
                    ),
                    "flow_checkpoint": args.flow_ckpt,
                    "rng_state_path": str(rng_state_path.resolve()),
                    "seed": sample_seed,
                    "sampler": {
                        "name": type(pipeline.sparse_structure_sampler).__name__,
                        "sigma_min": sigma_min,
                        "steps": args.steps,
                        "cfg_strength": args.cfg_strength,
                        "cfg_interval": list(args.cfg_interval),
                        "rescale_t": args.rescale_t,
                    },
                    "image_cond_model": image_cond_model,
                    "z_s": latent_stats(z_s[0]),
                }
                write_json(stats_path, stats)
                rows.append({**base_row, "skipped": False, "failed": False, "error": ""})
                del result, z_s, noise, cond
                print(f"[{sample_index + 1}/{len(records)}] sampled {sample_id}", flush=True)
            except Exception as exc:
                failure = {
                    **base_row,
                    "skipped": False,
                    "failed": True,
                    "error": repr(exc),
                }
                rows.append(failure)
                print(f"[{sample_index + 1}/{len(records)}] failed {sample_id}: {exc!r}", flush=True)
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="ss_flow",
                        args=args,
                        rows=rows,
                        extra_summary={
                            "sampler": {
                                "sigma_min": sigma_min,
                                **sampler_params,
                            },
                            "image_cond_model": image_cond_model,
                        },
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="ss_flow",
        args=args,
        rows=rows,
        extra_summary={
            "sampler": {
                "name": type(pipeline.sparse_structure_sampler).__name__,
                "sigma_min": sigma_min,
                "steps": args.steps,
                "cfg_strength": args.cfg_strength,
                "cfg_interval": list(args.cfg_interval),
                "rescale_t": args.rescale_t,
            },
            "image_cond_model": image_cond_model,
            "latent_space": config.get("dataset", {}).get("args", {}).get("latent_model"),
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
