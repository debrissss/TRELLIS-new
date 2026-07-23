#!/usr/bin/env python3
"""Stable3DGen-aligned mesh decoder loading and PLY export helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def add_stable3dgen_to_path(root: Path = STABLE3DGEN_ROOT) -> None:
    """Import Stable3DGen modules before TRELLIS modules with the same names."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Stable3DGen root not found: {root}")
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stable3dgen_mesh_decoder(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build the Stable3DGen mesh decoder using TRELLIS config args.

    TRELLIS fine-tuning config may name the model `ElasticSLatMeshDecoder`.
    The user confirmed it is structurally identical to Stable3DGen's
    `SLatMeshDecoder`, so we map the class name but keep strict checkpoint load.
    """
    add_stable3dgen_to_path()
    from hi3dgen.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder

    spec = config["models"]["decoder"]
    name = spec["name"]
    if name not in {"SLatMeshDecoder", "ElasticSLatMeshDecoder"}:
        raise ValueError(f"Unsupported decoder class for Stable3DGen mesh eval: {name}")
    decoder = SLatMeshDecoder(**spec["args"]).to(device)
    decoder.eval()
    return decoder


def load_decoder_checkpoint(decoder: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    """Strictly load a TRELLIS .pt decoder checkpoint into Stable3DGen decoder."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict in {checkpoint}, got {type(state)!r}")
    incompatible = decoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed for {checkpoint}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def rotate_mesh_x_positive_90(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Match Stable3DGen cli.py final export orientation."""
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2,
        direction=[1, 0, 0],
        point=[0, 0, 0],
    )
    mesh.apply_transform(rotation)
    return mesh


def export_stable3dgen_mesh(mesh_result: Any, output_path: Path) -> trimesh.Trimesh:
    """Export a Stable3DGen MeshExtractResult as PLY using final inference logic."""
    if not getattr(mesh_result, "success", False):
        raise RuntimeError("Mesh decoder returned an empty/unsuccessful MeshExtractResult")
    trimesh_mesh = mesh_result.to_trimesh(transform_pose=True)
    rotate_mesh_x_positive_90(trimesh_mesh)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mesh.export(output_path)
    reloaded = trimesh.load(output_path, force="mesh", process=False)
    if reloaded.vertices.shape[0] == 0 or reloaded.faces.shape[0] == 0:
        raise RuntimeError(f"Exported mesh is empty after reload: {output_path}")
    if not np.isfinite(reloaded.vertices).all():
        raise RuntimeError(f"Exported mesh contains non-finite vertices: {output_path}")
    return reloaded


def make_stable_sparse_tensor(coords: np.ndarray, feats: np.ndarray, device: torch.device, dtype: torch.dtype):
    """Create a Stable3DGen sparse tensor for a single latent sample."""
    add_stable3dgen_to_path()
    from hi3dgen.modules import sparse as sp

    coords_t = torch.as_tensor(coords, dtype=torch.int32, device=device)
    feats_t = torch.as_tensor(feats, dtype=dtype, device=device)
    if coords_t.ndim != 2 or coords_t.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N, 3), got {tuple(coords_t.shape)}")
    if feats_t.ndim != 2 or feats_t.shape[0] != coords_t.shape[0]:
        raise ValueError(f"Latent coords/feats mismatch: {tuple(coords_t.shape)} vs {tuple(feats_t.shape)}")
    batch = torch.zeros((coords_t.shape[0], 1), dtype=torch.int32, device=device)
    return sp.SparseTensor(coords=torch.cat([batch, coords_t], dim=-1), feats=feats_t)


def decode_latent_to_mesh_result(
    decoder: torch.nn.Module,
    latent_path: Path,
    device: torch.device,
) -> Any:
    """Load one latent npz and decode it with a Stable3DGen mesh decoder."""
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    with np.load(latent_path, allow_pickle=False) as data:
        for key in ("coords", "feats"):
            if key not in data.files:
                raise KeyError(f"{latent_path} missing key {key!r}; keys={data.files}")
        coords = data["coords"]
        feats = data["feats"]

    dtype = next(decoder.parameters()).dtype
    slat = make_stable_sparse_tensor(coords, feats, device=device, dtype=dtype)
    with torch.no_grad():
        results = decoder(slat)
    if len(results) != 1:
        raise RuntimeError(f"Expected one mesh result from single latent, got {len(results)}")
    result = results[0]
    if not torch.isfinite(result.vertices).all():
        raise RuntimeError(f"Decoded mesh contains non-finite vertices: {latent_path}")
    if not torch.isfinite(result.faces.float()).all():
        raise RuntimeError(f"Decoded mesh contains non-finite faces: {latent_path}")
    return result

