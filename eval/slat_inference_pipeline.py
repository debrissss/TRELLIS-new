#!/usr/bin/env python3
"""Orchestrate independent SLat encoder, flow, and decoder processes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import write_json
from eval.common.ss_inference import value_to_jsonable
from eval.ss_inference_pipeline import (
    _append_flag,
    _resolve_checkpoint,
    _run_stage,
    _stage_failure_count,
)


DEFAULT_STABLE3DGEN_ROOT = Path("/root/autodl-tmp/Stable3DGen")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["reconstruction", "generation", "all"],
        default="all",
        help="reconstruction=encoder->decoder; generation=SS coords->flow->decoder",
    )
    parser.add_argument("--output_root", type=Path, required=True)

    encoder_inputs = parser.add_mutually_exclusive_group()
    encoder_inputs.add_argument("--data_dir", type=Path)
    encoder_inputs.add_argument("--input_manifest", type=Path)
    parser.add_argument("--encoder_config", type=Path, default=None)
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--feature_model", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--sample_posterior", action="store_true")

    parser.add_argument(
        "--ss_manifest",
        type=Path,
        default=None,
        help="SS decoder manifest consumed by SLat Flow",
    )
    parser.add_argument("--flow_config", type=Path, default=None)
    parser.add_argument("--flow_ckpt", default=None)
    parser.add_argument("--stable3dgen_root", type=Path, default=DEFAULT_STABLE3DGEN_ROOT)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--preprocess_resolution", type=int, default=1024)
    parser.add_argument("--ignore_prepared_condition", action="store_true")
    parser.add_argument("--no_preprocess_image", dest="preprocess_image", action="store_false")
    parser.set_defaults(preprocess_image=True)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--decoder_config", type=Path, required=True)
    parser.add_argument("--decoder_ckpt", required=True)
    parser.add_argument(
        "--decoder_format",
        choices=["auto", "mesh", "gaussian"],
        default="auto",
    )

    parser.add_argument("--num_samples", type=int, default=16, help="<=0 selects all valid rows")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _includes_reconstruction(mode: str) -> bool:
    return mode in {"reconstruction", "all"}


def _includes_generation(mode: str) -> bool:
    return mode in {"generation", "all"}


def _require_args(args: argparse.Namespace) -> None:
    if _includes_reconstruction(args.mode):
        if args.encoder_config is None:
            raise ValueError(f"--encoder_config is required for mode={args.mode}")
        if not args.encoder_ckpt:
            raise ValueError(f"--encoder_ckpt is required for mode={args.mode}")
        if (args.data_dir is None) == (args.input_manifest is None):
            raise ValueError(
                f"Exactly one of --data_dir or --input_manifest is required for mode={args.mode}"
            )
    if _includes_generation(args.mode):
        if args.ss_manifest is None:
            raise ValueError(f"--ss_manifest is required for mode={args.mode}")
        if args.flow_config is None:
            raise ValueError(f"--flow_config is required for mode={args.mode}")
        if not args.flow_ckpt:
            raise ValueError(f"--flow_ckpt is required for mode={args.mode}")


def _resolve_optional_path(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


def _decoder_command(
    args: argparse.Namespace,
    *,
    latent_manifest: Path,
    output_dir: Path,
    common_flags: list[str],
) -> list[str]:
    command = [
        args.python,
        "-m",
        "eval.slat_decoder_inference",
        "--config",
        str(args.decoder_config),
        "--decoder_ckpt",
        _resolve_checkpoint(args.decoder_ckpt),
        "--latent_manifest",
        str(latent_manifest),
        "--output_dir",
        str(output_dir),
        "--format",
        args.decoder_format,
        "--device",
        args.device,
        *common_flags,
    ]
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_args(args)
    args.data_dir = _resolve_optional_path(args.data_dir)
    args.input_manifest = _resolve_optional_path(args.input_manifest)
    args.encoder_config = _resolve_optional_path(args.encoder_config)
    args.ss_manifest = _resolve_optional_path(args.ss_manifest)
    args.flow_config = _resolve_optional_path(args.flow_config)
    args.decoder_config = args.decoder_config.expanduser().resolve()
    args.stable3dgen_root = args.stable3dgen_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    encoder_dir = args.output_root / "encoder"
    flow_dir = args.output_root / "flow"
    decoder_encoder_dir = args.output_root / "decoder" / "from_encoder"
    decoder_flow_dir = args.output_root / "decoder" / "from_flow"

    common_flags: list[str] = []
    _append_flag(common_flags, args.skip_existing, "--skip_existing")
    _append_flag(common_flags, args.fail_on_error, "--fail_on_error")

    stage_specs: list[tuple[str, list[str], Path]] = []
    if _includes_reconstruction(args.mode):
        assert args.encoder_config is not None
        assert args.encoder_ckpt is not None
        encoder_command = [
            args.python,
            "-m",
            "eval.slat_encoder_inference",
            "--config",
            str(args.encoder_config),
            "--encoder_ckpt",
            _resolve_checkpoint(args.encoder_ckpt),
            "--output_dir",
            str(encoder_dir),
            "--num_samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            *common_flags,
        ]
        if args.data_dir is not None:
            encoder_command.extend(["--data_dir", str(args.data_dir)])
        else:
            assert args.input_manifest is not None
            encoder_command.extend(["--input_manifest", str(args.input_manifest)])
        if args.feature_model is not None:
            encoder_command.extend(["--feature_model", args.feature_model])
        if args.resolution is not None:
            encoder_command.extend(["--resolution", str(args.resolution)])
        if args.indices is not None:
            encoder_command.extend(["--indices", args.indices])
        _append_flag(encoder_command, args.sample_posterior, "--sample_posterior")
        stage_specs.append(("encoder", encoder_command, encoder_dir))

        decoder_encoder_command = _decoder_command(
            args,
            latent_manifest=encoder_dir / "manifest.csv",
            output_dir=decoder_encoder_dir,
            common_flags=common_flags,
        )
        stage_specs.append(
            ("decoder_from_encoder", decoder_encoder_command, decoder_encoder_dir)
        )

    if _includes_generation(args.mode):
        assert args.ss_manifest is not None
        assert args.flow_config is not None
        assert args.flow_ckpt is not None
        flow_command = [
            args.python,
            "-m",
            "eval.slat_flow_inference",
            "--config",
            str(args.flow_config),
            "--flow_ckpt",
            _resolve_checkpoint(args.flow_ckpt),
            "--ss_manifest",
            str(args.ss_manifest),
            "--output_dir",
            str(flow_dir),
            "--stable3dgen_root",
            str(args.stable3dgen_root),
            "--num_samples",
            str(args.num_samples),
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
        if args.indices is not None:
            flow_command.extend(["--indices", args.indices])
        _append_flag(
            flow_command,
            args.ignore_prepared_condition,
            "--ignore_prepared_condition",
        )
        _append_flag(flow_command, not args.preprocess_image, "--no_preprocess_image")
        _append_flag(flow_command, args.verbose, "--verbose")
        stage_specs.append(("flow", flow_command, flow_dir))

        decoder_flow_command = _decoder_command(
            args,
            latent_manifest=flow_dir / "manifest.csv",
            output_dir=decoder_flow_dir,
            common_flags=common_flags,
        )
        stage_specs.append(("decoder_from_flow", decoder_flow_command, decoder_flow_dir))

    run_manifest_path = args.output_root / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "args": value_to_jsonable(vars(args)),
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
