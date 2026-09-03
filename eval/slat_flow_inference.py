#!/usr/bin/env python3
"""Run image-conditioned structured-latent flow as an independent stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_csv, write_json
from eval.common.slat_inference import (
    denormalize_slat_feats,
    load_ss_coords,
    normalization_from_config,
    require_nonempty_unique_samples,
    save_slat_latent,
    slat_stats,
    successful_ss_coords_rows,
)
from eval.common.ss_inference import (
    load_image_condition,
    load_configured_model,
    load_torch_cpu_rng_state,
    require_device,
    sample_output_dir,
    write_stage_result,
)


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SLat flow training config JSON")
    parser.add_argument(
        "--flow_ckpt",
        required=True,
        help="Denoiser .pt checkpoint or from_pretrained-compatible model prefix",
    )
    parser.add_argument(
        "--ss_manifest",
        type=Path,
        required=True,
        help="manifest.csv produced by ss_decoder_inference",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stable3dgen_root",
        type=Path,
        default=DEFAULT_STABLE3DGEN_ROOT,
        help="Current Stable3DGen checkout providing image conditioning and sampler",
    )
    parser.add_argument("--num_samples", type=int, default=0, help="<=0 consumes all SS rows")
    parser.add_argument("--indices", default=None, help="Comma-separated indices in successful SS rows")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument(
        "--ignore_prepared_condition",
        action="store_true",
        help="Ignore SS Flow cond.png and preprocess the original condition again",
    )
    parser.add_argument(
        "--no_preprocess_image",
        dest="preprocess_image",
        action="store_false",
        help="Pass a raw selected condition directly to DINO when no prepared image is reused",
    )
    parser.set_defaults(preprocess_image=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def _build_flow_pipeline(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: Any,
):
    from eval.ss_flow_inference import _load_hi3dgen_components

    Hi3DGenPipeline, stable_samplers = _load_hi3dgen_components(
        args.stable3dgen_root
    )
    from hi3dgen import models as stable_models

    flow_model = load_configured_model(
        config,
        model_key="denoiser",
        checkpoint=args.flow_ckpt,
        device=device,
        models_module=stable_models,
        model_aliases={"ElasticSLatFlowModel": "SLatFlowModel"},
    )
    trainer_args = config.get("trainer", {}).get("args", {})
    sigma_min = float(trainer_args.get("sigma_min", 1e-5))
    image_cond_model = trainer_args.get("image_cond_model", "dinov2_vitl14_reg")
    sampler = stable_samplers.FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    # Minimal Stable3DGen pipeline: SLat Flow plus DINO only. SS components and
    # every SLat decoder remain absent from this process.
    pipeline = Hi3DGenPipeline(
        models={"slat_flow_model": flow_model},
        sparse_structure_sampler=None,
        slat_sampler=sampler,
        slat_normalization=None,
        image_cond_model=image_cond_model,
    )
    pipeline.slat_sampler_params = {}
    pipeline.to(device)
    return pipeline, sigma_min, image_cond_model


def _select_condition(
    record: dict[str, Any],
    *,
    ignore_prepared: bool,
) -> tuple[Path, bool]:
    if not ignore_prepared and bool(record["condition_preprocessed"]):
        return Path(record["selected_condition_path"]), True
    original_text = str(record.get("condition_image_path", "")).strip()
    if original_text:
        original_path = Path(original_text)
        if original_path.is_file():
            return original_path, False
    selected_path = Path(record["selected_condition_path"])
    if selected_path.is_file():
        return selected_path, False
    raise FileNotFoundError(
        f"{record['sample_id']}: no condition image is available for SLat Flow"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    import torch
    from PIL import Image

    from eval.ss_flow_inference import _preprocess_with_stable3dgen

    config = load_json(args.config)
    flow_spec = config.get("models", {}).get("denoiser")
    if flow_spec is None:
        raise KeyError("Config is missing models.denoiser")
    resolution = int(flow_spec["args"]["resolution"])
    in_channels = int(flow_spec["args"]["in_channels"])
    records = successful_ss_coords_rows(
        args.ss_manifest,
        num_samples=args.num_samples,
        seed=args.seed,
        indices=args.indices,
    )
    require_nonempty_unique_samples(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selected_samples.csv", records)

    device = require_device(args.device)
    pipeline, sigma_min, image_cond_model = _build_flow_pipeline(args, config, device)
    flow_model = pipeline.models["slat_flow_model"]
    normalization = normalization_from_config(config)
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
            coords_path = Path(record["coords_path"])
            source_seed = str(record.get("seed", "")).strip()
            sample_seed = int(source_seed) if source_seed else args.seed + sample_index
            condition_features_text = str(
                record.get("condition_features_path", "")
            ).strip()
            condition_features_path = (
                Path(condition_features_text) if condition_features_text else None
            )
            rng_state_text = str(record.get("rng_state_path", "")).strip()
            rng_state_path = Path(rng_state_text) if rng_state_text else None
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            latent_path = sample_dir / "latent.npz"
            prepared_path = sample_dir / "cond.png"
            stats_path = sample_dir / "stats.json"
            base_row = {
                "stage": "slat_flow",
                "sample_id": sample_id,
                "sample_index": sample_index,
                "dataset_index": record["dataset_index"],
                "ss_manifest": str(args.ss_manifest),
                "coords_path": str(coords_path),
                "condition_image_path": record.get("condition_image_path", ""),
                "condition_features_path": (
                    str(condition_features_path) if condition_features_path else ""
                ),
                "prepared_condition_path": str(prepared_path.resolve()),
                "latent_path": str(latent_path.resolve()),
                "latent_keys": "coords,feats,normalized_feats",
                "latent_domain": "decoder_ready",
                "rng_state_path": str(rng_state_path) if rng_state_path else "",
                "seed": sample_seed,
                "checkpoint": args.flow_ckpt,
            }
            if (
                args.skip_existing
                and latent_path.is_file()
                and prepared_path.is_file()
                and stats_path.is_file()
            ):
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                coords = load_ss_coords(coords_path, resolution=resolution)
                selected_condition, condition_preprocessed = _select_condition(
                    record,
                    ignore_prepared=args.ignore_prepared_condition,
                )
                with Image.open(selected_condition) as source_image:
                    image = source_image.convert("RGBA")
                if condition_preprocessed or not args.preprocess_image:
                    prepared = image
                else:
                    prepared = _preprocess_with_stable3dgen(
                        pipeline,
                        image,
                        resolution=args.preprocess_resolution,
                        stable3dgen_root=args.stable3dgen_root.expanduser().resolve(),
                    )

                if condition_features_path is not None:
                    cond = load_image_condition(
                        condition_features_path,
                        device=device,
                    )
                    condition_source = "ss_flow_artifact"
                else:
                    cond = pipeline.get_cond([prepared])
                    condition_source = "dino_fallback"
                if rng_state_path is not None:
                    torch.set_rng_state(load_torch_cpu_rng_state(rng_state_path))
                    rng_state_source = "ss_flow_artifact"
                else:
                    # Backward-compatible fallback for manifests produced before
                    # RNG handoff was added. New full-pipeline runs always use
                    # the exact post-SS state above.
                    torch.manual_seed(sample_seed)
                    rng_state_source = "seed_fallback"
                noise_feats = torch.randn(
                    coords.shape[0],
                    in_channels,
                ).to(device)
                from eval.stable3dgen_mesh_export import make_stable_sparse_tensor

                noise = make_stable_sparse_tensor(
                    coords,
                    noise_feats,
                    device=device,
                    dtype=noise_feats.dtype,
                )
                result = pipeline.slat_sampler.sample(
                    flow_model,
                    noise,
                    **cond,
                    **sampler_params,
                )
                normalized_slat = result["samples"]
                decoder_feats = denormalize_slat_feats(
                    normalized_slat.feats,
                    normalization,
                )
                output_coords = normalized_slat.coords[:, 1:]
                save_slat_latent(
                    latent_path,
                    coords=output_coords,
                    feats=decoder_feats,
                    normalized_feats=normalized_slat.feats,
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                prepared.save(prepared_path)
                stats = {
                    "stage": "slat_flow",
                    "sample_id": sample_id,
                    "ss_coords_path": str(coords_path),
                    "selected_condition_path": str(selected_condition),
                    "reused_preprocessed_condition": condition_preprocessed,
                    "condition_features_path": (
                        str(condition_features_path)
                        if condition_features_path
                        else None
                    ),
                    "condition_source": condition_source,
                    "prepared_condition_path": str(prepared_path.resolve()),
                    "flow_checkpoint": args.flow_ckpt,
                    "rng_state_path": str(rng_state_path) if rng_state_path else None,
                    "rng_state_source": rng_state_source,
                    "seed": sample_seed,
                    "sampler": {
                        "name": type(pipeline.slat_sampler).__name__,
                        "sigma_min": sigma_min,
                        "steps": args.steps,
                        "cfg_strength": args.cfg_strength,
                        "cfg_interval": list(args.cfg_interval),
                        "rescale_t": args.rescale_t,
                    },
                    "image_cond_model": image_cond_model,
                    "normalization_applied": normalization is not None,
                    "latent_domain": "decoder_ready",
                    "normalized_slat": slat_stats(
                        output_coords,
                        normalized_slat.feats,
                    ),
                    "decoder_ready_slat": slat_stats(
                        output_coords,
                        decoder_feats,
                    ),
                }
                write_json(stats_path, stats)
                rows.append({**base_row, "skipped": False, "failed": False, "error": ""})
                del result, normalized_slat, decoder_feats, noise, noise_feats, cond
                print(f"[{sample_index + 1}/{len(records)}] sampled {sample_id}", flush=True)
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "skipped": False,
                        "failed": True,
                        "error": repr(exc),
                    }
                )
                print(
                    f"[{sample_index + 1}/{len(records)}] failed {sample_id}: {exc!r}",
                    flush=True,
                )
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="slat_flow",
                        args=args,
                        rows=rows,
                        extra_summary={
                            "sampler": {
                                "sigma_min": sigma_min,
                                **sampler_params,
                            },
                            "image_cond_model": image_cond_model,
                            "normalization_applied": normalization is not None,
                            "latent_domain": "decoder_ready",
                        },
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="slat_flow",
        args=args,
        rows=rows,
        extra_summary={
            "source_ss_manifest": str(args.ss_manifest),
            "sampler": {
                "name": type(pipeline.slat_sampler).__name__,
                "sigma_min": sigma_min,
                "steps": args.steps,
                "cfg_strength": args.cfg_strength,
                "cfg_interval": list(args.cfg_interval),
                "rescale_t": args.rescale_t,
            },
            "image_cond_model": image_cond_model,
            "normalization_applied": normalization is not None,
            "latent_domain": "decoder_ready",
            "latent_model": config.get("dataset", {}).get("args", {}).get("latent_model"),
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
