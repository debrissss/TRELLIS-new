#!/usr/bin/env python3
"""评估已有 SLat Flow 生成产物，不执行采样或神经网络解码。"""

# 中文说明：
# gs-image 模式比较 flow 生成目录中的 generated_grid.png 与 gt_grid.png。
# mesh 模式比较 mesh 生成目录中的三角网格与 data_dir 中的 GT mesh。
# 本脚本只消费已经存在的产物，不加载 flow checkpoint 或 mesh decoder checkpoint。

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


def add_runs_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--runs", nargs="+", required=True, help=help_text)


def process_gs_image_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """评估一个或多个 flow run 的 GS decoder 图像结果。"""
    from eval.common.impl.slat_flow_gs_image_evaluation_impl import (
        compare_generation_runs,
    )

    return compare_generation_runs(
        parse_name_path_specs(args.runs),
        args.output_dir,
        skip_lpips=args.skip_lpips,
    )


def process_mesh_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """评估一个或多个已解码 mesh run 的几何结果。"""
    from eval.common.impl.slat_flow_mesh_evaluation_impl import (
        compare_mesh_runs_to_gt,
    )

    return compare_mesh_runs_to_gt(
        runs=parse_name_path_specs(args.runs),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        point_samples=args.point_samples,
        seed=args.seed,
        require_all_samples=args.require_all_samples,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gs_image = subparsers.add_parser(
        "gs-image",
        help="Evaluate generated GS decoder images against GT decoder images.",
    )
    add_runs_arg(
        gs_image,
        "One or more NAME=flow_generation_dir entries.",
    )
    gs_image.add_argument("--output_dir", type=Path, required=True)
    gs_image.add_argument("--skip_lpips", action="store_true")

    mesh = subparsers.add_parser(
        "mesh",
        help="Evaluate existing triangle mesh generation runs against GT meshes.",
    )
    add_runs_arg(
        mesh,
        "One or more NAME=mesh_generation_run_dir entries.",
    )
    mesh.add_argument("--data_dir", type=Path, required=True)
    mesh.add_argument("--output_dir", type=Path, required=True)
    mesh.add_argument("--point_samples", type=int, default=50000)
    mesh.add_argument("--seed", type=int, default=0)
    mesh.add_argument("--require_all_samples", action="store_true")

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "gs-image":
        result = process_gs_image_evaluation(args)
    else:
        result = process_mesh_evaluation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
