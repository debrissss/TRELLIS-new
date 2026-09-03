#!/usr/bin/env python3
"""Orchestrate independent SS encoder, flow, and decoder processes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import load_json, write_json
from eval.common.ss_inference import (
    read_input_manifest,
    require_nonempty_unique_samples,
    select_dataset_inputs,
    value_to_jsonable,
    write_input_manifest,
)


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["reconstruction", "generation", "all"],
        default="all",
        help="reconstruction=encoder->decoder; generation=flow->decoder",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data_dir", type=Path)
    inputs.add_argument("--input_manifest", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--vae_config", type=Path, required=True)
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--flow_config", type=Path, default=None)
    parser.add_argument("--flow_ckpt", default=None)
    parser.add_argument("--decoder_ckpt", required=True)
    parser.add_argument("--stable3dgen_root", type=Path, default=DEFAULT_STABLE3DGEN_ROOT)
    parser.add_argument("--num_samples", type=int, default=16, help="<=0 selects all valid samples")
    parser.add_argument("--indices", default=None, help="Comma-separated indices in the valid-sample list")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument("--no_preprocess_image", dest="preprocess_image", action="store_false")
    parser.set_defaults(preprocess_image=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a complete stage process when its manifest.csv already exists",
    )
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _includes_reconstruction(mode: str) -> bool:
    return mode in {"reconstruction", "all"}


def _includes_generation(mode: str) -> bool:
    return mode in {"generation", "all"}


def _require_args(args: argparse.Namespace) -> None:
    if _includes_reconstruction(args.mode) and not args.encoder_ckpt:
        raise ValueError(f"--encoder_ckpt is required for mode={args.mode}")
    if _includes_generation(args.mode):
        if args.flow_config is None:
            raise ValueError(f"--flow_config is required for mode={args.mode}")
        if not args.flow_ckpt:
            raise ValueError(f"--flow_ckpt is required for mode={args.mode}")


def _resolve_checkpoint(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    if Path(f"{path}.json").is_file() and Path(f"{path}.safetensors").is_file():
        return str(path.resolve())
    return value


def _append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _run_stage(
    *,
    name: str,
    command: list[str],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    if resume and manifest_path.is_file() and summary_path.is_file():
        print(f"[resume] skip {name}: {manifest_path}", flush=True)
        return {
            "name": name,
            "status": "skipped_existing_manifest",
            "command": command,
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
        }

    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError(
            f"Stage {name} completed without required manifest/summary: "
            f"{manifest_path}, {summary_path}"
        )
    return {
        "name": name,
        "status": "completed",
        "command": command,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }


def _stage_failure_count(stage: dict[str, Any]) -> int | None:
    summary_path = Path(stage["output_dir"]) / "summary.json"
    if not summary_path.is_file():
        return None
    return int(load_json(summary_path).get("failed_count", 0))


def _build_selection(args: argparse.Namespace, selection_path: Path) -> list[dict[str, Any]]:
    require_voxel = _includes_reconstruction(args.mode)
    require_condition = _includes_generation(args.mode)
    if args.resume and selection_path.is_file():
        return read_input_manifest(
            selection_path,
            require_voxel=require_voxel,
            require_condition=require_condition,
        )

    if args.input_manifest is not None:
        records = read_input_manifest(
            args.input_manifest,
            require_voxel=require_voxel,
            require_condition=require_condition,
        )
        write_input_manifest(selection_path, records)
        return records

    from eval.common.ss_inference import parse_indices

    assert args.data_dir is not None
    records = select_dataset_inputs(
        args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        indices=parse_indices(args.indices),
        require_voxel=require_voxel,
        require_condition=require_condition,
    )
    write_input_manifest(selection_path, records)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_args(args)
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
    if args.input_manifest is not None:
        args.input_manifest = args.input_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.vae_config = args.vae_config.expanduser().resolve()
    if args.flow_config is not None:
        args.flow_config = args.flow_config.expanduser().resolve()
    args.stable3dgen_root = args.stable3dgen_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    selection_path = args.output_root / "selected_samples.csv"
    records = _build_selection(args, selection_path)
    require_nonempty_unique_samples(records)

    encoder_dir = args.output_root / "encoder"
    flow_dir = args.output_root / "flow"
    decoder_encoder_dir = args.output_root / "decoder" / "from_encoder"
    decoder_flow_dir = args.output_root / "decoder" / "from_flow"

    common_flags: list[str] = []
    _append_flag(common_flags, args.skip_existing, "--skip_existing")
    _append_flag(common_flags, args.fail_on_error, "--fail_on_error")

    stage_specs: list[tuple[str, list[str], Path]] = []
    if _includes_reconstruction(args.mode):
        encoder_command = [
            args.python,
            "-m",
            "eval.ss_encoder_inference",
            "--config",
            str(args.vae_config),
            "--encoder_ckpt",
            _resolve_checkpoint(args.encoder_ckpt),
            "--input_manifest",
            str(selection_path),
            "--output_dir",
            str(encoder_dir),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            *common_flags,
        ]
        _append_flag(encoder_command, args.sample_posterior, "--sample_posterior")
        stage_specs.append(("encoder", encoder_command, encoder_dir))

        decoder_encoder_command = [
            args.python,
            "-m",
            "eval.ss_decoder_inference",
            "--config",
            str(args.vae_config),
            "--decoder_ckpt",
            _resolve_checkpoint(args.decoder_ckpt),
            "--latent_manifest",
            str(encoder_dir / "manifest.csv"),
            "--output_dir",
            str(decoder_encoder_dir),
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
            *common_flags,
        ]
        _append_flag(decoder_encoder_command, args.save_logits, "--save_logits")
        _append_flag(decoder_encoder_command, args.save_mesh, "--save_mesh")
        stage_specs.append(("decoder_from_encoder", decoder_encoder_command, decoder_encoder_dir))

    if _includes_generation(args.mode):
        assert args.flow_config is not None
        assert args.flow_ckpt is not None
        flow_command = [
            args.python,
            "-m",
            "eval.ss_flow_inference",
            "--config",
            str(args.flow_config),
            "--flow_ckpt",
            _resolve_checkpoint(args.flow_ckpt),
            "--stable3dgen_root",
            str(args.stable3dgen_root),
            "--input_manifest",
            str(selection_path),
            "--output_dir",
            str(flow_dir),
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
            "--preprocess_resolution",
            str(args.preprocess_resolution),
            "--device",
            args.device,
            *common_flags,
        ]
        _append_flag(flow_command, not args.preprocess_image, "--no_preprocess_image")
        _append_flag(flow_command, args.verbose, "--verbose")
        stage_specs.append(("flow", flow_command, flow_dir))

        decoder_flow_command = [
            args.python,
            "-m",
            "eval.ss_decoder_inference",
            "--config",
            str(args.vae_config),
            "--decoder_ckpt",
            _resolve_checkpoint(args.decoder_ckpt),
            "--latent_manifest",
            str(flow_dir / "manifest.csv"),
            "--output_dir",
            str(decoder_flow_dir),
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
            *common_flags,
        ]
        _append_flag(decoder_flow_command, args.save_logits, "--save_logits")
        _append_flag(decoder_flow_command, args.save_mesh, "--save_mesh")
        stage_specs.append(("decoder_from_flow", decoder_flow_command, decoder_flow_dir))

    run_manifest_path = args.output_root / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "args": value_to_jsonable(vars(args)),
        "selection_manifest": str(selection_path),
        "selected_samples": len(records),
        "planned_stages": [
            {
                "name": name,
                "command": command,
                "output_dir": str(output_dir),
            }
            for name, command, output_dir in stage_specs
        ],
        "stages": [],
    }
    write_json(run_manifest_path, run_manifest)

    try:
        for name, command, output_dir in stage_specs:
            stage = _run_stage(
                name=name,
                command=command,
                output_dir=output_dir,
                resume=args.resume,
            )
            run_manifest["stages"].append(stage)
            write_json(run_manifest_path, run_manifest)
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = repr(exc)
        write_json(run_manifest_path, run_manifest)
        raise

    failure_counts = [_stage_failure_count(stage) for stage in run_manifest["stages"]]
    known_failure_counts = [count for count in failure_counts if count is not None]
    run_manifest["failed_samples_across_stages"] = sum(known_failure_counts)
    run_manifest["status"] = (
        "completed_with_failures"
        if run_manifest["failed_samples_across_stages"]
        else "completed"
    )
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
