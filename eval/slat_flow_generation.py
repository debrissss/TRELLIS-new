#!/usr/bin/env python3
"""生成固定 SLat Flow 结果，或将已有生成 latent 解码成三角网格。"""

# 中文说明：
# flow 模式负责运行 SLat Flow，保存条件图、GS decoder 预览、PLY 和可选 latent NPZ。
# mesh 模式只把已有 generated_latent.npz 解码成三角网格，不计算任何 GT 指标。
# 所有指标计算统一由 eval/slat_flow_evaluation.py 完成。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import parse_name_path_specs
from eval.common.slat_flow import parse_mesh_decoder_specs


def add_flow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated dataset indices.",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")


def add_mesh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more NAME=flow_generation_dir entries.",
    )
    parser.add_argument("--mesh_config", type=Path, required=True)
    parser.add_argument("--mesh_decoder_ckpt", type=Path, required=True)
    parser.add_argument(
        "--run_mesh_decoders",
        nargs="+",
        default=None,
        help=(
            "Optional NAME=MESH_CONFIG=MESH_DECODER_CKPT overrides for runs "
            "that use different latent spaces."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_denormalize", action="store_true")
    parser.add_argument("--skip_existing_meshes", action="store_true")
    parser.add_argument("--require_all_samples", action="store_true")


def process_flow_generation(args: argparse.Namespace) -> dict[str, object]:
    """处理单个 flow checkpoint 的固定样本生成。"""
    from eval.common.impl.slat_flow_generation_impl import generate_fixed_samples

    return generate_fixed_samples(args)


def process_mesh_generation(args: argparse.Namespace) -> dict[str, object]:
    """批量将一个或多个 flow run 的 latent 解码成 mesh。"""
    from eval.common.impl.slat_flow_mesh_generation_impl import generate_flow_meshes

    return generate_flow_meshes(
        runs=parse_name_path_specs(args.runs),
        mesh_config_path=args.mesh_config,
        mesh_decoder_ckpt=args.mesh_decoder_ckpt,
        run_mesh_decoders=parse_mesh_decoder_specs(args.run_mesh_decoders),
        output_dir=args.output_dir,
        device_name=args.device,
        denormalize=not args.no_denormalize,
        skip_existing_meshes=args.skip_existing_meshes,
        require_all_samples=args.require_all_samples,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    flow = subparsers.add_parser("flow", help="Generate fixed SLat Flow samples.")
    add_flow_args(flow)
    mesh = subparsers.add_parser(
        "mesh",
        help="Decode existing generated SLat latents into triangle meshes.",
    )
    add_mesh_args(mesh)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "flow":
        result = process_flow_generation(args)
    else:
        result = process_mesh_generation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
