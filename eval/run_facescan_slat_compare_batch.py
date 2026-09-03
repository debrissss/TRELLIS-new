#!/usr/bin/env python3
"""Run the validated face-scan SS-replacement/SLat comparison for remaining IDs.

The script only orchestrates existing inference stages.  SS/SLat model forward
passes, sampling, decoding, image conditioning, mesh export, and triangle-mesh
voxelization remain implemented by the existing project/Stable3DGen code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_csv, write_json


FIRST_TESTED_ID = "2606090855170"
DEFAULT_DATA_ROOT = REPO_ROOT / "面扫测试数据"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/facescan_slat_flow_mesh_replace_compare/remaining_19"
)
DEFAULT_PYTHON = Path("/root/autodl-tmp/mamba_envs/trellis5090/bin/python")
DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")

WEIGHTS = {
    "ss_encoder": REPO_ROOT / "outputs/ss_weights/encoder_step0002000.pt",
    "ss_flow": REPO_ROOT / "outputs/ss_weights/denoiser_step0010000.pt",
    "ss_decoder": REPO_ROOT / "outputs/ss_weights/decoder_step0002000.pt",
    "slat_flow": REPO_ROOT
    / "outputs/train/slat_flow_enc_finetune_dec_finetune/ckpts/denoiser_step0015000.pt",
    "mesh_decoder": REPO_ROOT
    / "outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt",
    "gaussian_decoder": REPO_ROOT
    / "outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8/ckpts/decoder_step0004000.pt",
}
CONFIGS = {
    "ss": REPO_ROOT / "configs/vae/ss_vae_conv3d_16l8_fp16.json",
    "ss_flow": REPO_ROOT
    / "configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json",
    "slat_flow": REPO_ROOT
    / "outputs/train/slat_flow_enc_finetune_dec_finetune/config.json",
    "mesh_decoder": REPO_ROOT / "outputs/train/slat_dec_mesh_fine_tune/config.json",
    "gaussian_decoder": REPO_ROOT
    / "outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8/config.json",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stable3dgen-root", type=Path, default=DEFAULT_STABLE3DGEN_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ss-steps", type=int, default=50)
    parser.add_argument("--slat-steps", type=int, default=6)
    parser.add_argument("--render-resolution", type=int, default=768)
    parser.add_argument(
        "--ids",
        default=None,
        help="Optional comma-separated IDs. By default all complete IDs except the tested first ID are used.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--expected-count", type=int, default=19)
    parser.add_argument("--include-first", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute existing stage artifacts")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/copy/voxelize inputs and manifests without loading GPU models",
    )
    return parser


def required_inputs(data_root: Path, sample_id: str) -> dict[str, Path]:
    sample_root = data_root / sample_id
    return {
        "normal": sample_root / "model/up_normal.png",
        "merged_mesh": sample_root
        / "align_to_standard_filter/merged_normalized_mesh.ply",
        "model_geometry": sample_root / "model_normalized_nocolor.ply",
    }


def discover_ids(args: argparse.Namespace) -> list[str]:
    if args.ids:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    else:
        ids = sorted(
            item.name
            for item in args.data_root.iterdir()
            if item.is_dir() and item.name.isdigit()
        )
        if not args.include_first:
            ids = [item for item in ids if item != FIRST_TESTED_ID]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs were selected")
    incomplete: dict[str, list[str]] = {}
    for sample_id in ids:
        missing = [name for name, path in required_inputs(args.data_root, sample_id).items() if not path.is_file()]
        if missing:
            incomplete[sample_id] = missing
    if incomplete:
        raise FileNotFoundError(f"Incomplete input IDs: {incomplete}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        ids = ids[: args.limit]
    elif args.ids is None and args.expected_count > 0 and len(ids) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} remaining IDs, discovered {len(ids)}; "
            "use --expected-count 0 to accept the discovered count"
        )
    if not ids:
        raise ValueError("No input IDs selected")
    return ids


def copy_file(source: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.is_file():
        shutil.copy2(source, destination)


def ply_vertex_count(path: Path) -> int:
    import utils3d

    return int(len(utils3d.io.read_ply(path)[0]))


def prepare_inputs(args: argparse.Namespace, ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from dataset_toolkits.voxelize import _voxelize

    voxel_work = args.output_root / "voxelization_work"
    (voxel_work / "renders").mkdir(parents=True, exist_ok=True)
    (voxel_work / "voxels").mkdir(parents=True, exist_ok=True)
    ss_rows: list[dict[str, Any]] = []
    voxel_rows: list[dict[str, Any]] = []
    for dataset_index, sample_id in enumerate(ids):
        source = required_inputs(args.data_root, sample_id)
        input_root = args.output_root / "inputs" / sample_id
        copy_file(source["normal"], input_root / "up_normal.png", force=args.force)

        merged_id = f"{sample_id}__merged_filter"
        merged_input = input_root / "merged_filter/input_geometry/mesh.ply"
        merged_voxel = input_root / "merged_filter/voxel/voxelized.ply"
        copy_file(source["merged_mesh"], merged_input, force=args.force)
        toolkit_input = voxel_work / "renders" / merged_id / "mesh.ply"
        toolkit_voxel = voxel_work / "voxels" / f"{merged_id}.ply"
        copy_file(source["merged_mesh"], toolkit_input, force=args.force)
        if args.force or not toolkit_voxel.is_file():
            _voxelize(str(source["merged_mesh"]), merged_id, str(voxel_work))
        copy_file(toolkit_voxel, merged_voxel, force=True)

        model_id = f"{sample_id}__model_normalized"
        model_input = input_root / "model_normalized/input_geometry/mesh.ply"
        model_voxel = input_root / "model_normalized/voxel/voxelized.ply"
        copy_file(source["model_geometry"], model_input, force=args.force)
        model_toolkit_input = voxel_work / "renders" / model_id / "mesh.ply"
        model_toolkit_voxel = voxel_work / "voxels" / f"{model_id}.ply"
        copy_file(source["model_geometry"], model_toolkit_input, force=args.force)
        if args.force or not model_toolkit_voxel.is_file():
            _voxelize(str(source["model_geometry"]), model_id, str(voxel_work))
        copy_file(model_toolkit_voxel, model_voxel, force=True)

        ss_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "condition_image_path": str(source["normal"].resolve()),
            }
        )
        for branch_index, row in enumerate(
            [
                {
                    "sample_id": merged_id,
                    "geometry": "triangle_mesh",
                    "input": merged_input,
                    "voxel": merged_voxel,
                    "voxel_source": "dataset_toolkits/voxelize.py::_voxelize",
                },
                {
                    "sample_id": model_id,
                    "geometry": "triangle_mesh_without_vertex_color",
                    "input": model_input,
                    "voxel": model_voxel,
                    "voxel_source": "dataset_toolkits/voxelize.py::_voxelize",
                },
            ]
        ):
            voxel_rows.append(
                {
                    "base_id": sample_id,
                    "sample_id": row["sample_id"],
                    "dataset_index": dataset_index * 2 + branch_index,
                    "input_geometry": row["geometry"],
                    "input_geometry_path": str(row["input"].resolve()),
                    "voxel_path": str(row["voxel"].resolve()),
                    "voxel_source": row["voxel_source"],
                    "voxel_count": ply_vertex_count(row["voxel"]),
                    "condition_image_path": str(source["normal"].resolve()),
                }
            )
    return ss_rows, voxel_rows


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("EGL_PLATFORM", "surfaceless")
    return env


def run_command(
    args: argparse.Namespace,
    stage: str,
    command: list[str],
    commands: list[dict[str, Any]],
) -> None:
    log_path = args.output_root / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{stage}] {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=command_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    record = {
        "stage": stage,
        "command": command,
        "log": str(log_path.resolve()),
        "return_code": return_code,
    }
    commands.append(record)
    write_json(args.output_root / "command_history.json", commands)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def inference_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if not args.force:
        flags.append("--skip_existing")
    if args.fail_fast:
        flags.append("--fail_on_error")
    return flags


def build_handoff(args: argparse.Namespace, voxel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared_flow = args.output_root / "shared_ss_reference/flow/samples"
    ss_decoder = args.output_root / "ss_mesh_reconstruction/decoder/from_encoder/samples"
    handoff: list[dict[str, Any]] = []
    for row in voxel_rows:
        base_id = row["base_id"]
        sample_id = row["sample_id"]
        shared = shared_flow / base_id
        coords = ss_decoder / sample_id / "coords.npz"
        required = [
            coords,
            shared / "cond.png",
            shared / "condition_features.npz",
            shared / "rng_state.npz",
        ]
        if not all(path.is_file() for path in required):
            print(f"[handoff] skip incomplete {sample_id}: {[str(p) for p in required if not p.is_file()]}")
            continue
        handoff.append(
            {
                "sample_id": sample_id,
                "dataset_index": row["dataset_index"],
                "coords_path": str(coords.resolve()),
                "condition_image_path": row["condition_image_path"],
                "prepared_condition_path": str((shared / "cond.png").resolve()),
                "condition_preprocessed": True,
                "condition_features_path": str((shared / "condition_features.npz").resolve()),
                "rng_state_path": str((shared / "rng_state.npz").resolve()),
                "seed": args.seed + int(row["dataset_index"]) // 2,
                "source_stage": "ss_encoder_decoder_replace",
                "failed": False,
            }
        )
    if not handoff:
        raise RuntimeError("No complete SS replacement artifacts are available for SLat Flow")
    return handoff


def successful_count(summary_path: Path) -> tuple[int, int]:
    if not summary_path.is_file():
        return 0, 1
    summary = load_json(summary_path)
    return int(summary.get("successful_count", 0)), int(summary.get("failed_count", 0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["sample_id"]: row for row in csv.DictReader(file) if row.get("sample_id")}


def render_results(
    args: argparse.Namespace,
    ids: list[str],
    commands: list[dict[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    mesh_samples = args.output_root / "slat_mesh_generation/decoder/from_flow/samples"
    for index, sample_id in enumerate(ids, start=1):
        output_dir = args.output_root / "renders" / sample_id
        output_path = output_dir / "comparison_3x2.png"
        if output_path.is_file() and not args.force:
            print(f"[render {index}/{len(ids)}] skip existing {output_path}")
            continue
        inputs = required_inputs(args.data_root, sample_id)
        command = [
            str(args.python),
            "-m",
            "eval.render_facescan_slat_compare",
            "--sample-id",
            sample_id,
            "--normal",
            str(inputs["normal"]),
            "--merged-source",
            str(inputs["merged_mesh"]),
            "--merged-output",
            str(mesh_samples / f"{sample_id}__merged_filter/mesh.ply"),
            "--model-source",
            str(inputs["model_geometry"]),
            "--model-output",
            str(mesh_samples / f"{sample_id}__model_normalized/mesh.ply"),
            "--output-dir",
            str(output_dir),
            "--resolution",
            str(args.render_resolution),
        ]
        try:
            run_command(args, f"render_{sample_id}", command, commands)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})
            if args.fail_fast:
                raise
    return failures


def write_results(
    args: argparse.Namespace,
    ids: list[str],
    voxel_rows: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    render_failures: list[dict[str, str]],
) -> dict[str, Any]:
    ss_encoder = read_manifest(args.output_root / "ss_mesh_reconstruction/encoder/manifest.csv")
    ss_decoder = read_manifest(
        args.output_root / "ss_mesh_reconstruction/decoder/from_encoder/manifest.csv"
    )
    slat_flow = read_manifest(args.output_root / "slat_mesh_generation/flow/manifest.csv")
    mesh_decoder = read_manifest(
        args.output_root / "slat_mesh_generation/decoder/from_flow/manifest.csv"
    )
    gaussian_decoder = read_manifest(args.output_root / "slat_gaussian_decoder/manifest.csv")
    rows: list[dict[str, Any]] = []
    for voxel in voxel_rows:
        sample_id = voxel["sample_id"]
        base_id = voxel["base_id"]
        shared = args.output_root / "shared_ss_reference/flow/samples" / base_id
        rows.append(
            {
                **voxel,
                "condition_path": str((shared / "cond.png").resolve()),
                "condition_features_path": str((shared / "condition_features.npz").resolve()),
                "rng_state_path": str((shared / "rng_state.npz").resolve()),
                "ss_latent_path": ss_encoder.get(sample_id, {}).get("latent_path", ""),
                "ss_coords_path": ss_decoder.get(sample_id, {}).get("coords_path", ""),
                "slat_latent_path": slat_flow.get(sample_id, {}).get("latent_path", ""),
                "mesh_path": mesh_decoder.get(sample_id, {}).get("artifact_path", ""),
                "gaussian_path": gaussian_decoder.get(sample_id, {}).get("artifact_path", ""),
                "comparison_image": str(
                    (args.output_root / f"renders/{base_id}/comparison_3x2.png").resolve()
                ),
            }
        )
    write_csv(args.output_root / "results_manifest.csv", rows)

    stage_paths = {
        "shared_ss_flow": args.output_root / "shared_ss_reference/flow/summary.json",
        "shared_ss_decoder": args.output_root
        / "shared_ss_reference/decoder/from_flow/summary.json",
        "ss_encoder": args.output_root / "ss_mesh_reconstruction/encoder/summary.json",
        "ss_decoder": args.output_root
        / "ss_mesh_reconstruction/decoder/from_encoder/summary.json",
        "slat_flow": args.output_root / "slat_mesh_generation/flow/summary.json",
        "mesh_decoder": args.output_root
        / "slat_mesh_generation/decoder/from_flow/summary.json",
        "gaussian_decoder": args.output_root / "slat_gaussian_decoder/summary.json",
    }
    stages = {
        name: {"successful": successful_count(path)[0], "failed": successful_count(path)[1]}
        for name, path in stage_paths.items()
    }
    shared_checks: dict[str, Any] = {}
    for sample_id in ids:
        shared = args.output_root / "shared_ss_reference/flow/samples" / sample_id
        if all((shared / name).is_file() for name in ["cond.png", "condition_features.npz", "rng_state.npz"]):
            shared_checks[sample_id] = {
                "condition_sha256": sha256(shared / "cond.png"),
                "condition_features_sha256": sha256(shared / "condition_features.npz"),
                "rng_state_sha256": sha256(shared / "rng_state.npz"),
            }
    failed_total = sum(item["failed"] for item in stages.values()) + len(render_failures)
    summary = {
        "status": "completed" if failed_total == 0 else "completed_with_failures",
        "selected_ids": ids,
        "num_ids": len(ids),
        "num_branches": len(voxel_rows),
        "output_root": str(args.output_root.resolve()),
        "weights": {name: str(path.resolve()) for name, path in WEIGHTS.items()},
        "configs": {name: str(path.resolve()) for name, path in CONFIGS.items()},
        "stages": stages,
        "shared_artifact_checks": shared_checks,
        "render_failures": render_failures,
        "commands": commands,
        "results_manifest": str((args.output_root / "results_manifest.csv").resolve()),
    }
    write_json(args.output_root / "batch_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.stable3dgen_root = args.stable3dgen_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    required_files = [args.python, *WEIGHTS.values(), *CONFIGS.values()]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing runtime files: {missing}")
    if not args.stable3dgen_root.is_dir():
        raise FileNotFoundError(args.stable3dgen_root)

    ids = discover_ids(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    run_manifest = {
        "status": "preparing",
        "selected_ids": ids,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "commands": commands,
    }
    write_json(args.output_root / "batch_run_manifest.json", run_manifest)
    print(f"Selected {len(ids)} IDs: {', '.join(ids)}")

    ss_rows, voxel_rows = prepare_inputs(args, ids)
    write_csv(args.output_root / "shared_ss_input.csv", ss_rows)
    write_csv(args.output_root / "voxelization_manifest.csv", voxel_rows)
    write_csv(
        args.output_root / "ss_reconstruction_input.csv",
        [
            {
                "sample_id": row["sample_id"],
                "dataset_index": row["dataset_index"],
                "voxel_path": row["voxel_path"],
                "condition_image_path": row["condition_image_path"],
            }
            for row in voxel_rows
        ],
    )
    if args.prepare_only:
        run_manifest["status"] = "prepared"
        run_manifest["num_ids"] = len(ids)
        run_manifest["num_branches"] = len(voxel_rows)
        write_json(args.output_root / "batch_run_manifest.json", run_manifest)
        return run_manifest

    common = inference_flags(args)
    python = str(args.python)
    run_manifest["status"] = "running"
    write_json(args.output_root / "batch_run_manifest.json", run_manifest)
    try:
        run_command(
            args,
            "shared_ss_reference",
            [
                python, "-m", "eval.ss_inference_pipeline",
                "--mode", "generation",
                "--input_manifest", str(args.output_root / "shared_ss_input.csv"),
                "--output_root", str(args.output_root / "shared_ss_reference"),
                "--vae_config", str(CONFIGS["ss"]),
                "--flow_config", str(CONFIGS["ss_flow"]),
                "--flow_ckpt", str(WEIGHTS["ss_flow"]),
                "--decoder_ckpt", str(WEIGHTS["ss_decoder"]),
                "--stable3dgen_root", str(args.stable3dgen_root),
                "--num_samples", "0",
                "--seed", str(args.seed),
                "--device", args.device,
                "--steps", str(args.ss_steps),
                "--cfg_strength", "3",
                "--cfg_interval", "0", "1",
                "--rescale_t", "1",
                "--preprocess_resolution", "1024",
                "--threshold", "0",
                "--save_logits", "--save_mesh",
                *common,
            ],
            commands,
        )
        run_command(
            args,
            "ss_mesh_reconstruction",
            [
                python, "-m", "eval.ss_inference_pipeline",
                "--mode", "reconstruction",
                "--input_manifest", str(args.output_root / "ss_reconstruction_input.csv"),
                "--output_root", str(args.output_root / "ss_mesh_reconstruction"),
                "--vae_config", str(CONFIGS["ss"]),
                "--encoder_ckpt", str(WEIGHTS["ss_encoder"]),
                "--decoder_ckpt", str(WEIGHTS["ss_decoder"]),
                "--seed", str(args.seed),
                "--device", args.device,
                "--threshold", "0",
                "--save_logits", "--save_mesh",
                *common,
            ],
            commands,
        )
        handoff = build_handoff(args, voxel_rows)
        write_csv(args.output_root / "slat_handoff_manifest.csv", handoff)
        run_command(
            args,
            "slat_mesh_generation",
            [
                python, "-m", "eval.slat_inference_pipeline",
                "--mode", "generation",
                "--ss_manifest", str(args.output_root / "slat_handoff_manifest.csv"),
                "--output_root", str(args.output_root / "slat_mesh_generation"),
                "--flow_config", str(CONFIGS["slat_flow"]),
                "--flow_ckpt", str(WEIGHTS["slat_flow"]),
                "--decoder_config", str(CONFIGS["mesh_decoder"]),
                "--decoder_ckpt", str(WEIGHTS["mesh_decoder"]),
                "--decoder_format", "mesh",
                "--stable3dgen_root", str(args.stable3dgen_root),
                "--steps", str(args.slat_steps),
                "--cfg_strength", "3",
                "--cfg_interval", "0.5", "1",
                "--rescale_t", "3",
                "--preprocess_resolution", "1024",
                "--num_samples", "0",
                "--seed", str(args.seed),
                "--device", args.device,
                *common,
            ],
            commands,
        )
        run_command(
            args,
            "slat_gaussian_decoder",
            [
                python, "-m", "eval.slat_decoder_inference",
                "--config", str(CONFIGS["gaussian_decoder"]),
                "--decoder_ckpt", str(WEIGHTS["gaussian_decoder"]),
                "--latent_manifest", str(args.output_root / "slat_mesh_generation/flow/manifest.csv"),
                "--output_dir", str(args.output_root / "slat_gaussian_decoder"),
                "--format", "gaussian",
                "--device", args.device,
                *common,
            ],
            commands,
        )
        render_failures = [] if args.skip_render else render_results(args, ids, commands)
        summary = write_results(args, ids, voxel_rows, commands, render_failures)
        run_manifest["status"] = summary["status"]
        run_manifest["commands"] = commands
        run_manifest["summary"] = str((args.output_root / "batch_summary.json").resolve())
        write_json(args.output_root / "batch_run_manifest.json", run_manifest)
        return summary
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = repr(exc)
        run_manifest["commands"] = commands
        write_json(args.output_root / "batch_run_manifest.json", run_manifest)
        raise


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(f"\nBatch status: {summary['status']}")
    print(f"Output root: {args.output_root.expanduser().resolve()}")


if __name__ == "__main__":
    main()
