#!/usr/bin/env python3
"""Orchestrate the complete split image-to-3D generation chain."""

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
from eval.common.ss_inference import value_to_jsonable
from eval.ss_inference_pipeline import _append_flag, _resolve_checkpoint


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")
TERMINAL_PIPELINE_STATUSES = {"completed", "completed_with_failures"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data_dir", type=Path)
    inputs.add_argument("--input_manifest", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)

    parser.add_argument("--ss_vae_config", type=Path, required=True)
    parser.add_argument("--ss_flow_config", type=Path, required=True)
    parser.add_argument("--ss_flow_ckpt", required=True)
    parser.add_argument("--ss_decoder_ckpt", required=True)
    parser.add_argument("--ss_steps", type=int, default=50)
    parser.add_argument("--ss_cfg_strength", type=float, default=3.0)
    parser.add_argument("--ss_cfg_interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--ss_rescale_t", type=float, default=1.0)
    parser.add_argument("--ss_threshold", type=float, default=0.0)

    parser.add_argument("--slat_flow_config", type=Path, required=True)
    parser.add_argument("--slat_flow_ckpt", required=True)
    parser.add_argument("--slat_decoder_config", type=Path, required=True)
    parser.add_argument("--slat_decoder_ckpt", required=True)
    parser.add_argument(
        "--slat_decoder_format",
        choices=["auto", "mesh", "gaussian"],
        default="auto",
    )
    parser.add_argument("--slat_steps", type=int, default=6)
    parser.add_argument("--slat_cfg_strength", type=float, default=3.0)
    parser.add_argument("--slat_cfg_interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)

    parser.add_argument("--stable3dgen_root", type=Path, default=DEFAULT_STABLE3DGEN_ROOT)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument("--no_preprocess_image", dest="preprocess_image", action="store_false")
    parser.set_defaults(preprocess_image=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _run_pipeline(
    *,
    name: str,
    command: list[str],
    output_root: Path,
) -> dict[str, Any]:
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    child_manifest_path = output_root / "run_manifest.json"
    if not child_manifest_path.is_file():
        raise RuntimeError(
            f"Pipeline {name} completed without run manifest: {child_manifest_path}"
        )
    child_manifest = load_json(child_manifest_path)
    if child_manifest.get("status") not in TERMINAL_PIPELINE_STATUSES:
        raise RuntimeError(
            f"Pipeline {name} has non-terminal status "
            f"{child_manifest.get('status')!r}: {child_manifest_path}"
        )
    return {
        "name": name,
        "status": child_manifest["status"],
        "command": command,
        "output_root": str(output_root),
        "run_manifest": str(child_manifest_path),
        "failed_samples_across_stages": int(
            child_manifest.get("failed_samples_across_stages", 0)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data_dir = _resolve_path(args.data_dir) if args.data_dir is not None else None
    args.input_manifest = (
        _resolve_path(args.input_manifest) if args.input_manifest is not None else None
    )
    args.output_root = _resolve_path(args.output_root)
    args.ss_vae_config = _resolve_path(args.ss_vae_config)
    args.ss_flow_config = _resolve_path(args.ss_flow_config)
    args.slat_flow_config = _resolve_path(args.slat_flow_config)
    args.slat_decoder_config = _resolve_path(args.slat_decoder_config)
    args.stable3dgen_root = _resolve_path(args.stable3dgen_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    ss_root = args.output_root / "ss"
    slat_root = args.output_root / "slat"
    ss_decoder_manifest = ss_root / "decoder" / "from_flow" / "manifest.csv"
    final_manifest = slat_root / "decoder" / "from_flow" / "manifest.csv"

    shared_flags: list[str] = []
    _append_flag(shared_flags, args.skip_existing, "--skip_existing")
    _append_flag(shared_flags, args.resume, "--resume")
    _append_flag(shared_flags, args.fail_on_error, "--fail_on_error")

    ss_command = [
        args.python,
        "-m",
        "eval.ss_inference_pipeline",
        "--mode",
        "generation",
        "--output_root",
        str(ss_root),
        "--vae_config",
        str(args.ss_vae_config),
        "--flow_config",
        str(args.ss_flow_config),
        "--flow_ckpt",
        _resolve_checkpoint(args.ss_flow_ckpt),
        "--decoder_ckpt",
        _resolve_checkpoint(args.ss_decoder_ckpt),
        "--stable3dgen_root",
        str(args.stable3dgen_root),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.ss_steps),
        "--cfg_strength",
        str(args.ss_cfg_strength),
        "--cfg_interval",
        str(args.ss_cfg_interval[0]),
        str(args.ss_cfg_interval[1]),
        "--rescale_t",
        str(args.ss_rescale_t),
        "--preprocess_resolution",
        str(args.preprocess_resolution),
        "--threshold",
        str(args.ss_threshold),
        "--device",
        args.device,
        *shared_flags,
    ]
    if args.data_dir is not None:
        ss_command.extend(["--data_dir", str(args.data_dir)])
    else:
        assert args.input_manifest is not None
        ss_command.extend(["--input_manifest", str(args.input_manifest)])
    if args.indices is not None:
        ss_command.extend(["--indices", args.indices])
    _append_flag(ss_command, not args.preprocess_image, "--no_preprocess_image")
    _append_flag(ss_command, args.verbose, "--verbose")

    slat_command = [
        args.python,
        "-m",
        "eval.slat_inference_pipeline",
        "--mode",
        "generation",
        "--output_root",
        str(slat_root),
        "--ss_manifest",
        str(ss_decoder_manifest),
        "--flow_config",
        str(args.slat_flow_config),
        "--flow_ckpt",
        _resolve_checkpoint(args.slat_flow_ckpt),
        "--decoder_config",
        str(args.slat_decoder_config),
        "--decoder_ckpt",
        _resolve_checkpoint(args.slat_decoder_ckpt),
        "--decoder_format",
        args.slat_decoder_format,
        "--stable3dgen_root",
        str(args.stable3dgen_root),
        "--num_samples",
        "0",
        "--seed",
        str(args.seed),
        "--steps",
        str(args.slat_steps),
        "--cfg_strength",
        str(args.slat_cfg_strength),
        "--cfg_interval",
        str(args.slat_cfg_interval[0]),
        str(args.slat_cfg_interval[1]),
        "--rescale_t",
        str(args.slat_rescale_t),
        "--preprocess_resolution",
        str(args.preprocess_resolution),
        "--device",
        args.device,
        *shared_flags,
    ]
    _append_flag(slat_command, not args.preprocess_image, "--no_preprocess_image")
    _append_flag(slat_command, args.verbose, "--verbose")

    run_manifest_path = args.output_root / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "status": "running",
        "args": value_to_jsonable(vars(args)),
        "ss_decoder_manifest": str(ss_decoder_manifest),
        "final_manifest": str(final_manifest),
        "planned_pipelines": [
            {"name": "ss_generation", "command": ss_command, "output_root": str(ss_root)},
            {
                "name": "slat_generation",
                "command": slat_command,
                "output_root": str(slat_root),
            },
        ],
        "pipelines": [],
    }
    write_json(run_manifest_path, run_manifest)

    try:
        ss_result = _run_pipeline(
            name="ss_generation",
            command=ss_command,
            output_root=ss_root,
        )
        run_manifest["pipelines"].append(ss_result)
        write_json(run_manifest_path, run_manifest)
        if not ss_decoder_manifest.is_file():
            raise RuntimeError(
                f"SS pipeline did not produce the SLat handoff manifest: "
                f"{ss_decoder_manifest}"
            )

        slat_result = _run_pipeline(
            name="slat_generation",
            command=slat_command,
            output_root=slat_root,
        )
        run_manifest["pipelines"].append(slat_result)
        if not final_manifest.is_file():
            raise RuntimeError(
                f"SLat pipeline did not produce the final manifest: {final_manifest}"
            )
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = repr(exc)
        write_json(run_manifest_path, run_manifest)
        raise

    run_manifest["failed_samples_across_pipelines"] = sum(
        int(item["failed_samples_across_stages"])
        for item in run_manifest["pipelines"]
    )
    run_manifest["status"] = (
        "completed_with_failures"
        if run_manifest["failed_samples_across_pipelines"]
        else "completed"
    )
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
