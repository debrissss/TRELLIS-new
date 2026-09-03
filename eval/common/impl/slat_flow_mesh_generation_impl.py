"""将已有 SLat Flow latent 解码成三角网格，不计算评估指标。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval.common.io import load_json, safe_tag, write_csv, write_json
from eval.common.mesh_metrics import load_mesh
from eval.common.model_loading import (
    build_stable3dgen_mesh_decoder,
    export_stable3dgen_mesh,
    load_decoder_checkpoint,
    make_stable_sparse_tensor,
)
from eval.common.slat_flow import (
    latent_normalization_from_config,
    load_flow_config,
    load_flow_manifest,
    load_generated_latent,
    resolve_generated_latent_path,
)


def decode_latent_arrays_to_mesh(
    decoder,
    coords: np.ndarray,
    feats: np.ndarray,
    output_path: Path,
    device,
) -> Any:
    """用已加载的 mesh decoder 解码一条 latent 并导出 PLY。"""
    import torch

    dtype = next(decoder.parameters()).dtype
    slat = make_stable_sparse_tensor(coords, feats, device=device, dtype=dtype)
    with torch.no_grad():
        results = decoder(slat)
    if len(results) != 1:
        raise RuntimeError(f"Expected one mesh result from single latent, got {len(results)}")
    result = results[0]
    if not torch.isfinite(result.vertices).all():
        raise RuntimeError("Decoded mesh contains non-finite vertices")
    if not torch.isfinite(result.faces.float()).all():
        raise RuntimeError("Decoded mesh contains non-finite faces")
    return export_stable3dgen_mesh(result, output_path)


def mesh_artifact_stats(mesh) -> dict[str, Any]:
    """计算生成阶段所需的轻量完整性统计，不执行连通域或 GT 对比。"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    return {
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "extents": (bounds_max - bounds_min).tolist(),
    }


def generate_single_mesh_run(
    *,
    name: str,
    flow_run_dir: Path,
    mesh_config: dict[str, Any],
    mesh_decoder_ckpt: Path,
    run_output_dir: Path,
    device,
    denormalize: bool,
    skip_existing_meshes: bool,
    require_all_samples: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """解码一组 flow 生成 latent，输出独立 mesh 生成任务目录。"""
    flow_config = load_flow_config(flow_run_dir)
    normalization = latent_normalization_from_config(flow_config)
    decoder = build_stable3dgen_mesh_decoder(mesh_config, device)
    load_decoder_checkpoint(decoder, mesh_decoder_ckpt, device)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    mesh_dir = run_output_dir / "meshes"
    for sample_index, flow_row in enumerate(load_flow_manifest(flow_run_dir)):
        sample_id = flow_row["sample_id"]
        latent_path = resolve_generated_latent_path(flow_run_dir, flow_row)
        mesh_path = mesh_dir / f"{sample_id}.ply"
        row: dict[str, Any] = {
            "run": name,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "flow_run_dir": str(flow_run_dir),
            "generated_latent_path": str(latent_path),
            "mesh_path": str(mesh_path),
            "mesh_decoder_ckpt": str(mesh_decoder_ckpt),
            "denormalized": bool(denormalize),
            "failed": False,
            "error": "",
        }
        try:
            if str(flow_row.get("failed", "")).lower() in {"1", "true", "yes"}:
                raise RuntimeError(
                    f"Flow generation failed for {sample_id}: {flow_row.get('error', '')}"
                )
            if skip_existing_meshes and mesh_path.is_file():
                mesh = load_mesh(mesh_path)
            else:
                coords, feats = load_generated_latent(
                    latent_path,
                    normalization=normalization,
                    denormalize=denormalize,
                )
                mesh = decode_latent_arrays_to_mesh(
                    decoder,
                    coords,
                    feats,
                    mesh_path,
                    device,
                )
            row.update(mesh_artifact_stats(mesh))
            print(f"[mesh-generation][{name}][{sample_index + 1}] OK {sample_id}", flush=True)
        except Exception as exc:
            row.update({"failed": True, "error": repr(exc)})
            failures.append(row)
            print(
                f"[mesh-generation][{name}][{sample_index + 1}] "
                f"FAIL {sample_id}: {exc!r}",
                flush=True,
            )
            if require_all_samples:
                raise
        rows.append(row)

    successful = [row for row in rows if not row["failed"]]
    summary = {
        "run": name,
        "flow_run_dir": str(flow_run_dir),
        "output_dir": str(run_output_dir),
        "mesh_decoder_ckpt": str(mesh_decoder_ckpt),
        "denormalized": bool(denormalize),
        "num_records": len(rows),
        "successful_samples": len(successful),
        "failed_count": len(failures),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
    }
    write_csv(run_output_dir / "manifest.csv", rows)
    write_json(run_output_dir / "summary.json", summary)

    del decoder
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()
    return summary, rows


def generate_flow_meshes(
    *,
    runs: dict[str, Path],
    mesh_config_path: Path,
    mesh_decoder_ckpt: Path,
    run_mesh_decoders: dict[str, tuple[Path, Path]] | None,
    output_dir: Path,
    device_name: str,
    denormalize: bool = True,
    skip_existing_meshes: bool = False,
    require_all_samples: bool = False,
) -> dict[str, Any]:
    """批量解码多个 flow run；每个 run 有独立产物目录和 manifest。"""
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_mesh_decoders = run_mesh_decoders or {}
    unknown_overrides = sorted(set(run_mesh_decoders) - set(runs))
    if unknown_overrides:
        raise ValueError(f"Mesh decoder overrides have no matching run: {unknown_overrides}")

    summaries: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for name, flow_run_dir in runs.items():
        run_config_path, run_ckpt = run_mesh_decoders.get(
            name,
            (mesh_config_path, mesh_decoder_ckpt),
        )
        run_output_dir = output_dir / safe_tag(name)
        summary, rows = generate_single_mesh_run(
            name=name,
            flow_run_dir=flow_run_dir,
            mesh_config=load_json(run_config_path),
            mesh_decoder_ckpt=run_ckpt,
            run_output_dir=run_output_dir,
            device=device,
            denormalize=denormalize,
            skip_existing_meshes=skip_existing_meshes,
            require_all_samples=require_all_samples,
        )
        summaries[name] = summary
        all_rows.extend(rows)

    result = {
        "runs": summaries,
        "output_dir": str(output_dir),
        "mesh_config": str(mesh_config_path),
        "mesh_decoder_ckpt": str(mesh_decoder_ckpt),
        "run_mesh_decoders": {
            name: {
                "mesh_config": str(config_path),
                "mesh_decoder_ckpt": str(checkpoint_path),
            }
            for name, (config_path, checkpoint_path) in run_mesh_decoders.items()
        },
        "num_records": len(all_rows),
        "failed_count": sum(1 for row in all_rows if row["failed"]),
        "denormalized": bool(denormalize),
    }
    write_csv(output_dir / "all_runs_summary.csv", list(summaries.values()))
    write_json(output_dir / "summary.json", result)
    return result
