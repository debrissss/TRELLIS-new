#!/usr/bin/env python3
"""Run one structured-latent decoder as an independent inference stage."""

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

from eval.common.io import load_json, write_json
from eval.common.slat_inference import (
    load_slat_latent,
    make_trellis_sparse_tensor,
    slat_stats,
    successful_slat_rows,
)
from eval.common.ss_inference import (
    load_configured_model,
    require_device,
    sample_output_dir,
    write_stage_result,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SLat decoder config JSON")
    parser.add_argument(
        "--decoder_ckpt",
        required=True,
        help="Decoder .pt checkpoint or from_pretrained-compatible model prefix",
    )
    parser.add_argument(
        "--latent_manifest",
        type=Path,
        required=True,
        help="manifest.csv produced by slat_encoder_inference or slat_flow_inference",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=["auto", "mesh", "gaussian"],
        default="auto",
        help="auto infers the representation from models.decoder.name",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def _decoder_format(config: dict[str, Any], requested: str) -> str:
    decoder_spec = config.get("models", {}).get("decoder")
    if decoder_spec is None:
        raise KeyError("Config is missing models.decoder")
    decoder_name = decoder_spec["name"].lower()
    inferred = (
        "mesh"
        if "mesh" in decoder_name
        else "gaussian"
        if "gaussian" in decoder_name
        else None
    )
    if requested == "auto":
        if inferred is None:
            raise ValueError(
                "Only mesh and Gaussian SLat decoder artifacts are currently supported; "
                f"cannot infer from {decoder_spec['name']!r}"
            )
        return inferred
    if inferred is not None and requested != inferred:
        raise ValueError(
            f"--format {requested!r} does not match decoder class {decoder_spec['name']!r}"
        )
    return requested


def _checkpoint_prefix_exists(checkpoint: str) -> bool:
    path = Path(checkpoint).expanduser()
    return Path(f"{path}.json").is_file() and Path(f"{path}.safetensors").is_file()


def _load_mesh_decoder(
    config: dict[str, Any],
    checkpoint: str,
    device: Any,
):
    """Reuse the current Stable3DGen-aligned mesh decoder loading path."""

    from eval.stable3dgen_mesh_export import (
        add_stable3dgen_to_path,
        build_stable3dgen_mesh_decoder,
        load_decoder_checkpoint,
    )

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        decoder = build_stable3dgen_mesh_decoder(config, device)
        load_decoder_checkpoint(decoder, checkpoint_path, device)
        return decoder
    if checkpoint_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        raise FileNotFoundError(checkpoint_path)
    if not (_checkpoint_prefix_exists(checkpoint) or "/" in checkpoint):
        raise FileNotFoundError(
            f"Checkpoint is neither a .pt file nor a pretrained prefix: {checkpoint}"
        )

    add_stable3dgen_to_path()
    from hi3dgen import models as stable_models

    decoder = stable_models.from_pretrained(checkpoint).to(device)
    decoder.eval()
    return decoder


def _gaussian_stats(representation: Any) -> dict[str, Any]:
    xyz = representation.get_xyz.detach().float().cpu()
    return {
        "num_gaussians": int(xyz.shape[0]),
        "bounds_min": xyz.min(dim=0).values.tolist(),
        "bounds_max": xyz.max(dim=0).values.tolist(),
        "opacity_mean": float(representation.get_opacity.detach().float().mean().cpu()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    import torch

    config = load_json(args.config)
    output_format = _decoder_format(config, args.format)
    producer_rows = successful_slat_rows(args.latent_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(args.device)

    if output_format == "mesh":
        decoder = _load_mesh_decoder(
            config,
            args.decoder_ckpt,
            device,
        )
    else:
        decoder = load_configured_model(
            config,
            model_key="decoder",
            checkpoint=args.decoder_ckpt,
            device=device,
        )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_index, producer_row in enumerate(producer_rows):
            sample_id = producer_row["sample_id"]
            latent_path = Path(producer_row["latent_path"])
            sample_dir = sample_output_dir(args.output_dir, sample_id)
            artifact_path = sample_dir / (
                "mesh.ply" if output_format == "mesh" else "gaussian.ply"
            )
            stats_path = sample_dir / "stats.json"
            source_stage = producer_row.get("stage", "unknown")
            base_row = {
                "stage": "slat_decoder",
                "source_stage": source_stage,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "latent_path": str(latent_path),
                "latent_domain": producer_row.get("latent_domain", ""),
                "condition_image_path": producer_row.get("condition_image_path", ""),
                "prepared_condition_path": producer_row.get(
                    "prepared_condition_path", ""
                ),
                "decoder_format": output_format,
                "artifact_path": str(artifact_path.resolve()),
                "stats_path": str(stats_path.resolve()),
                "checkpoint": args.decoder_ckpt,
            }
            if (
                args.skip_existing
                and artifact_path.is_file()
                and stats_path.is_file()
            ):
                rows.append({**base_row, "skipped": True, "failed": False, "error": ""})
                continue
            try:
                coords, feats = load_slat_latent(latent_path)
                sample_dir.mkdir(parents=True, exist_ok=True)
                if output_format == "mesh":
                    from eval.common.impl.slat_flow_mesh_generation_impl import (
                        decode_latent_arrays_to_mesh,
                        mesh_artifact_stats,
                    )

                    representation = decode_latent_arrays_to_mesh(
                        decoder,
                        coords,
                        feats,
                        artifact_path,
                        device,
                    )
                    representation_stats = mesh_artifact_stats(representation)
                else:
                    latent = make_trellis_sparse_tensor(
                        coords,
                        feats,
                        device=device,
                    )
                    representations = decoder(latent)
                    if len(representations) != 1:
                        raise RuntimeError(
                            "Expected one Gaussian representation from one latent, "
                            f"got {len(representations)}"
                        )
                    representation = representations[0]
                    representation.save_ply(artifact_path)
                    representation_stats = _gaussian_stats(representation)
                    del latent, representations

                stats = {
                    "stage": "slat_decoder",
                    "source_stage": source_stage,
                    "sample_id": sample_id,
                    "latent_path": str(latent_path),
                    "latent_domain": producer_row.get("latent_domain", ""),
                    "decoder_checkpoint": args.decoder_ckpt,
                    "decoder_format": output_format,
                    "latent": slat_stats(coords, feats),
                    "artifact_path": str(artifact_path.resolve()),
                    "representation": representation_stats,
                }
                write_json(stats_path, stats)
                rows.append(
                    {
                        **base_row,
                        **representation_stats,
                        "skipped": False,
                        "failed": False,
                        "error": "",
                    }
                )
                print(f"[{sample_index + 1}/{len(producer_rows)}] decoded {sample_id}", flush=True)
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
                    f"[{sample_index + 1}/{len(producer_rows)}] failed {sample_id}: {exc!r}",
                    flush=True,
                )
                if args.fail_on_error:
                    write_stage_result(
                        args.output_dir,
                        stage="slat_decoder",
                        args=args,
                        rows=rows,
                        extra_summary={
                            "source_manifest": str(args.latent_manifest),
                            "decoder_format": output_format,
                        },
                    )
                    raise

    return write_stage_result(
        args.output_dir,
        stage="slat_decoder",
        args=args,
        rows=rows,
        extra_summary={
            "source_manifest": str(args.latent_manifest),
            "decoder_format": output_format,
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
