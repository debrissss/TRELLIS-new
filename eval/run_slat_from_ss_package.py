#!/usr/bin/env python3
"""Continue at SLat Flow from an exported SS-stage safetensors package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from safetensors import safe_open
from safetensors.torch import load_file

from trellis import models
from trellis.pipelines import samplers
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", type=Path)
    parser.add_argument("--flow-config", type=Path)
    parser.add_argument("--flow-ckpt", type=Path)
    parser.add_argument("--mesh-config", type=Path)
    parser.add_argument("--mesh-ckpt", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=5.0)
    parser.add_argument("--cfg-interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_training_model(config_path: Path, checkpoint: Path, model_key: str):
    config_path = config_path.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["models"][model_key]
    model = getattr(models, model_spec["name"])(**model_spec["args"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model, config


def build_slat_pipeline(
    args: argparse.Namespace,
) -> tuple[TrellisImageTo3DPipeline, dict]:
    training_paths = (
        args.flow_config,
        args.flow_ckpt,
        args.mesh_config,
        args.mesh_ckpt,
    )
    if args.deploy_dir is not None and any(path is not None for path in training_paths):
        raise ValueError("Use either --deploy-dir or the four training weight arguments")
    if args.deploy_dir is None and not all(path is not None for path in training_paths):
        raise ValueError(
            "Provide --deploy-dir, or all of --flow-config/--flow-ckpt/"
            "--mesh-config/--mesh-ckpt"
        )

    if args.deploy_dir is not None:
        deploy_dir = args.deploy_dir.expanduser().resolve()
        config = json.loads(
            (deploy_dir / "pipeline.json").read_text(encoding="utf-8")
        )
        pipeline_args = config["args"]
        model_paths = pipeline_args["models"]
        slat_models = {
            "slat_flow_model": models.from_pretrained(
                str(deploy_dir / model_paths["slat_flow_model"])
            ),
            "slat_decoder_mesh": models.from_pretrained(
                str(deploy_dir / model_paths["slat_decoder_mesh"])
            ),
        }
        normalization = pipeline_args["slat_normalization"]
        provenance = {
            "kind": "deploy",
            "deploy_dir": str(deploy_dir),
            "flow": str(deploy_dir / model_paths["slat_flow_model"]),
            "mesh_decoder": str(deploy_dir / model_paths["slat_decoder_mesh"]),
        }
    else:
        assert args.flow_config is not None and args.flow_ckpt is not None
        assert args.mesh_config is not None and args.mesh_ckpt is not None
        flow_model, flow_config = _load_training_model(
            args.flow_config, args.flow_ckpt, "denoiser"
        )
        mesh_decoder, _ = _load_training_model(
            args.mesh_config, args.mesh_ckpt, "decoder"
        )
        slat_models = {
            "slat_flow_model": flow_model,
            "slat_decoder_mesh": mesh_decoder,
        }
        normalization = flow_config["dataset"]["args"]["normalization"]
        provenance = {
            "kind": "training_checkpoints",
            "flow_config": str(args.flow_config.expanduser().resolve()),
            "flow_checkpoint": str(args.flow_ckpt.expanduser().resolve()),
            "flow_checkpoint_sha256": sha256(args.flow_ckpt.expanduser().resolve()),
            "mesh_config": str(args.mesh_config.expanduser().resolve()),
            "mesh_checkpoint": str(args.mesh_ckpt.expanduser().resolve()),
            "mesh_checkpoint_sha256": sha256(args.mesh_ckpt.expanduser().resolve()),
        }

    # Construct only the SLat half. Image features and SS coordinates are
    # already immutable tensors in the handoff package, so DINO and SS models
    # must not be loaded or re-executed here.
    pipeline = TrellisImageTo3DPipeline()
    pipeline.models = slat_models
    for model in pipeline.models.values():
        model.eval()
    pipeline.slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    pipeline.slat_sampler_params = {
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "cfg_interval": list(args.cfg_interval),
        "rescale_t": args.rescale_t,
    }
    pipeline.slat_normalization = normalization
    pipeline.to(torch.device(args.device))
    return pipeline, provenance


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    package_path = args.package.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(package_path), device="cpu")
    required = {"coords", "cond", "neg_cond"}
    missing = required.difference(tensors)
    if missing:
        raise KeyError(f"SS package is missing tensors: {sorted(missing)}")
    coords = tensors["coords"]
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise ValueError(f"Invalid coords shape: {tuple(coords.shape)}")
    if coords[:, 0].count_nonzero().item() != 0:
        raise ValueError("Only batch-zero SS packages are supported")
    if coords[:, 1:].min().item() < 0 or coords[:, 1:].max().item() >= 64:
        raise ValueError("SS coordinates must be within resolution 64")
    if tensors["cond"].shape != tensors["neg_cond"].shape:
        raise ValueError("cond and neg_cond shapes differ")

    with safe_open(package_path, framework="pt", device="cpu") as file:
        package_metadata = file.metadata() or {}
    pipeline, model_provenance = build_slat_pipeline(args)
    device = pipeline.device
    cond = {
        "cond": tensors["cond"].to(device=device),
        "neg_cond": tensors["neg_cond"].to(device=device),
    }
    coords = coords.to(device=device, dtype=torch.int32)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        slat = pipeline.sample_slat(cond, coords)
        mesh_result = pipeline.decode_slat(slat, formats=["mesh"])["mesh"][0]
    if not mesh_result.success:
        raise RuntimeError("SLat mesh decoder returned an empty mesh")

    latent_path = output_dir / "slat_latent.npz"
    np.savez_compressed(
        latent_path,
        coords=slat.coords.detach().cpu().numpy(),
        feats=slat.feats.detach().float().cpu().numpy(),
    )
    mesh_path = output_dir / "slat_generated_mesh.ply"
    mesh = trimesh.Trimesh(
        vertices=mesh_result.vertices.detach().float().cpu().numpy(),
        faces=mesh_result.faces.detach().long().cpu().numpy(),
        process=False,
    )
    mesh.export(mesh_path)

    manifest = {
        "stage": "slat_flow_and_mesh_decode",
        "ss_package": str(package_path),
        "ss_package_sha256": sha256(package_path),
        "ss_package_metadata": package_metadata,
        "model_provenance": model_provenance,
        "loaded_model_keys": sorted(pipeline.models),
        "sampler": {
            "seed": args.seed,
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
            "cfg_interval": list(args.cfg_interval),
            "rescale_t": args.rescale_t,
        },
        "ss_coords": int(coords.shape[0]),
        "slat_shape": list(slat.feats.shape),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "latent_path": str(latent_path),
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256(mesh_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
