#!/usr/bin/env python3
"""Run the unsplit Stable3DGen normal-image path and save parity artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal_image", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stable3dgen_root", type=Path, default=DEFAULT_STABLE3DGEN_ROOT)

    parser.add_argument("--ss_flow_config", type=Path, required=True)
    parser.add_argument("--ss_flow_ckpt", type=Path, required=True)
    parser.add_argument("--ss_decoder_config", type=Path, required=True)
    parser.add_argument("--ss_decoder_ckpt", type=Path, required=True)
    parser.add_argument("--slat_flow_config", type=Path, required=True)
    parser.add_argument("--slat_flow_ckpt", type=Path, required=True)
    parser.add_argument("--slat_decoder_config", type=Path, required=True)
    parser.add_argument("--slat_decoder_ckpt", type=Path, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ss_steps", type=int, default=50)
    parser.add_argument("--ss_cfg_strength", type=float, default=3.0)
    parser.add_argument("--ss_cfg_interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--ss_rescale_t", type=float, default=1.0)
    parser.add_argument("--slat_steps", type=int, default=6)
    parser.add_argument("--slat_cfg_strength", type=float, default=3.0)
    parser.add_argument("--slat_cfg_interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _unwrap_state_dict(state: Any, checkpoint: Path) -> dict[str, Any]:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {checkpoint}")
    return state


def _load_stable_model(
    stable_models: Any,
    config: dict[str, Any],
    *,
    model_key: str,
    checkpoint: Path,
    device: Any,
):
    import torch

    spec = config.get("models", {}).get(model_key)
    if spec is None:
        raise KeyError(f"Config is missing models.{model_key}")
    aliases = {
        "ElasticSLatFlowModel": "SLatFlowModel",
        "ElasticSLatMeshDecoder": "SLatMeshDecoder",
    }
    model_name = aliases.get(spec["name"], spec["name"])
    model = getattr(stable_models, model_name)(**spec["args"])
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    incompatible = model.load_state_dict(
        _unwrap_state_dict(state, checkpoint),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed for {checkpoint}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    model = model.to(device)
    model.eval()
    return model


def _preprocess_normal(pipeline: Any, image: Any, *, resolution: int, root: Path):
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        return pipeline.preprocess_image(image, resolution=resolution)
    finally:
        os.chdir(previous_cwd)


def _array_stats(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    import torch
    from PIL import Image

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    device = torch.device(args.device)

    args.normal_image = _resolve(args.normal_image)
    args.output_dir = _resolve(args.output_dir)
    args.stable3dgen_root = _resolve(args.stable3dgen_root)
    for name in (
        "ss_flow_config",
        "ss_flow_ckpt",
        "ss_decoder_config",
        "ss_decoder_ckpt",
        "slat_flow_config",
        "slat_flow_ckpt",
        "slat_decoder_config",
        "slat_decoder_ckpt",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root_text = str(args.stable3dgen_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    from hi3dgen import models as stable_models
    from hi3dgen.pipelines import Hi3DGenPipeline, samplers

    ss_flow_config = load_json(args.ss_flow_config)
    ss_decoder_config = load_json(args.ss_decoder_config)
    slat_flow_config = load_json(args.slat_flow_config)
    slat_decoder_config = load_json(args.slat_decoder_config)

    models = {
        "sparse_structure_flow_model": _load_stable_model(
            stable_models,
            ss_flow_config,
            model_key="denoiser",
            checkpoint=args.ss_flow_ckpt,
            device=device,
        ),
        "sparse_structure_decoder": _load_stable_model(
            stable_models,
            ss_decoder_config,
            model_key="decoder",
            checkpoint=args.ss_decoder_ckpt,
            device=device,
        ),
        "slat_flow_model": _load_stable_model(
            stable_models,
            slat_flow_config,
            model_key="denoiser",
            checkpoint=args.slat_flow_ckpt,
            device=device,
        ),
        "slat_decoder_mesh": _load_stable_model(
            stable_models,
            slat_decoder_config,
            model_key="decoder",
            checkpoint=args.slat_decoder_ckpt,
            device=device,
        ),
    }

    ss_trainer_args = ss_flow_config.get("trainer", {}).get("args", {})
    slat_trainer_args = slat_flow_config.get("trainer", {}).get("args", {})
    image_cond_model = ss_trainer_args.get(
        "image_cond_model",
        slat_trainer_args.get("image_cond_model", "dinov2_vitl14_reg"),
    )
    normalization = slat_flow_config.get("dataset", {}).get("args", {}).get(
        "normalization"
    )
    if normalization is None:
        raise KeyError("SLat flow config is missing dataset.args.normalization")

    pipeline = Hi3DGenPipeline(
        models=models,
        sparse_structure_sampler=samplers.FlowEulerGuidanceIntervalSampler(
            sigma_min=float(ss_trainer_args.get("sigma_min", 1e-5))
        ),
        slat_sampler=samplers.FlowEulerGuidanceIntervalSampler(
            sigma_min=float(slat_trainer_args.get("sigma_min", 1e-5))
        ),
        slat_normalization=normalization,
        image_cond_model=image_cond_model,
    )
    pipeline.sparse_structure_sampler_params = {}
    pipeline.slat_sampler_params = {}
    pipeline.to(device)

    with Image.open(args.normal_image) as source:
        input_image = source.convert("RGBA")
    prepared = _preprocess_normal(
        pipeline,
        input_image,
        resolution=args.preprocess_resolution,
        root=args.stable3dgen_root,
    )
    prepared_path = args.output_dir / "cond.png"
    prepared.save(prepared_path)

    captured: dict[str, Any] = {}

    def capture_ss_latent(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["z_s"] = inputs[0].detach().float().cpu()

    hook = models["sparse_structure_decoder"].register_forward_pre_hook(
        capture_ss_latent
    )
    try:
        with torch.no_grad():
            cond = pipeline.get_cond([prepared])
            condition_path = args.output_dir / "condition_features.npz"
            np.savez_compressed(
                condition_path,
                cond=cond["cond"].detach().float().cpu().numpy(),
                neg_cond=cond["neg_cond"].detach().float().cpu().numpy(),
            )
            # This is the exact ordering used by Hi3DGenPipeline.run().
            torch.manual_seed(args.seed)
            coords = pipeline.sample_sparse_structure(
                cond,
                1,
                {
                    "steps": args.ss_steps,
                    "cfg_strength": args.ss_cfg_strength,
                    "cfg_interval": tuple(args.ss_cfg_interval),
                    "rescale_t": args.ss_rescale_t,
                },
            )
            rng_state_after_ss = torch.get_rng_state()
            slat = pipeline.sample_slat(
                cond,
                coords,
                {
                    "steps": args.slat_steps,
                    "cfg_strength": args.slat_cfg_strength,
                    "cfg_interval": tuple(args.slat_cfg_interval),
                    "rescale_t": args.slat_rescale_t,
                },
            )
            decoded = pipeline.decode_slat(slat, ["mesh"])
    finally:
        hook.remove()

    if "z_s" not in captured:
        raise RuntimeError("Stable3DGen SS decoder hook did not capture z_s")
    z_s = captured["z_s"][0].numpy()
    coords_array = coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    slat_coords = (
        slat.coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    )
    slat_feats = slat.feats.detach().float().cpu().numpy()
    mean = np.asarray(normalization["mean"], dtype=np.float32)[None]
    std = np.asarray(normalization["std"], dtype=np.float32)[None]
    normalized_feats = (slat_feats - mean) / std

    ss_latent_path = args.output_dir / "ss_latent.npz"
    ss_coords_path = args.output_dir / "ss_coords.npz"
    rng_state_path = args.output_dir / "rng_state_after_ss.npz"
    slat_latent_path = args.output_dir / "slat_latent.npz"
    np.savez_compressed(ss_latent_path, z_s=z_s)
    np.savez_compressed(ss_coords_path, coords=coords_array)
    np.savez_compressed(
        rng_state_path,
        torch_cpu_rng_state=rng_state_after_ss.cpu().numpy().astype(np.uint8),
    )
    np.savez_compressed(
        slat_latent_path,
        coords=slat_coords,
        feats=slat_feats,
        normalized_feats=normalized_feats,
    )

    from eval.stable3dgen_mesh_export import export_stable3dgen_mesh

    mesh_path = args.output_dir / "mesh.ply"
    mesh = export_stable3dgen_mesh(decoded["mesh"][0], mesh_path)
    summary = {
        "stage": "stable3dgen_reference",
        "normal_image": str(args.normal_image),
        "prepared_condition_path": str(prepared_path),
        "condition_features_path": str(condition_path),
        "seed": args.seed,
        "ss_latent_path": str(ss_latent_path),
        "ss_coords_path": str(ss_coords_path),
        "rng_state_path": str(rng_state_path),
        "slat_latent_path": str(slat_latent_path),
        "mesh_path": str(mesh_path),
        "ss_latent": _array_stats(z_s),
        "ss_coords_count": int(coords_array.shape[0]),
        "slat_feats": _array_stats(slat_feats),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
