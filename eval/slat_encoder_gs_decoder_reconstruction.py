#!/usr/bin/env python3
"""Evaluate SLat encoder + Gaussian decoder reconstruction."""

# 中文说明：
# 评估 SLat encoder + Gaussian/GS decoder 的重建质量。
# single 模式加载 encoder_ckpt 和 decoder_ckpt，渲染固定视角图像并计算图像重建指标。
# many/compare 模式只汇总已有 summary.json，不重新跑模型前向。

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
from eval.common.summary import compare_summary_dirs


def process_single_run(args: argparse.Namespace) -> dict[str, object]:
    from eval.common.impl.slat_encoder_gs_decoder_reconstruction_impl import evaluate

    return evaluate(args)


def add_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--encoder_ckpt", type=Path, required=True)
    parser.add_argument("--decoder_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--view_indices", default="0", help="Comma-separated fixed view indices, e.g. 0 or 0,4,8,12.")
    parser.add_argument("--feature_model", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_images", type=int, default=16)
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")


def process_many_summaries(args: argparse.Namespace) -> list[dict[str, object]]:
    return compare_summary_dirs(parse_name_path_specs(args.runs), args.output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0].startswith("-"):
        parser = argparse.ArgumentParser(description=__doc__)
        add_single_args(parser)
        args = parser.parse_args(argv)
        args.command = "single"
        return args

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single", help="Evaluate one encoder/decoder pair.")
    add_single_args(single)
    many = subparsers.add_parser("many", help="Compare existing single-run summary.json files.")
    many.add_argument("--runs", nargs="+", required=True, help="One or more NAME=eval_output_dir entries.")
    many.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="Alias of many.")
    compare.add_argument("--runs", nargs="+", required=True, help="One or more NAME=eval_output_dir entries.")
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "single":
        print(json.dumps(process_single_run(args), indent=2, ensure_ascii=False))
        return
    rows = process_many_summaries(args)
    print(f"[OK] wrote comparison CSV: {args.output} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
