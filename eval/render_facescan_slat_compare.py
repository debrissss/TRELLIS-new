#!/usr/bin/env python3
"""Render the first face-scan SLat mesh-replacement comparison as a 3x2 grid.

Mesh rendering, camera setup, labels, and Open3D options are reused from
``vis.render_mesh_compare``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from vis.render_mesh_compare import (
    RenderOptions,
    draw_label,
    load_mesh,
    render_mesh,
)


FRONT_VIEW = {"eye_axis": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--merged-source", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--row2-input-label", default="Merged-filter input mesh")
    parser.add_argument("--row2-output-label", default="Merged-filter SLat output")
    parser.add_argument(
        "--row3-input-label", default="Model-normalized no-color input mesh"
    )
    parser.add_argument("--row3-output-label", default="Model-normalized SLat output")
    parser.add_argument("--resolution", type=int, default=768)
    return parser.parse_args()


def geometry_points(geometry: object) -> np.ndarray:
    if isinstance(geometry, o3d.geometry.PointCloud):
        return np.asarray(geometry.points, dtype=np.float64)
    return np.asarray(geometry.vertices, dtype=np.float64)


def shared_frame(geometries: list[object], options: RenderOptions) -> tuple[np.ndarray, float, float]:
    points = np.concatenate([geometry_points(item) for item in geometries], axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    eye = np.asarray(FRONT_VIEW["eye_axis"], dtype=np.float64)
    up = np.asarray(FRONT_VIEW["up"], dtype=np.float64)
    forward = -eye
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    projected = np.stack(((points - center) @ right, (points - center) @ up), axis=1)
    projected_min = projected.min(axis=0)
    projected_max = projected.max(axis=0)
    projected_center = (projected_min + projected_max) * 0.5
    visible_pixels = options.resolution - options.margin_pixels * 2
    half_extent = (
        float(np.max(projected_max - projected_min))
        * 0.5
        * options.resolution
        / visible_pixels
        * options.padding
    )
    target = center + right * projected_center[0] + up * projected_center[1]
    depth_extent = float(np.max(bbox_max - bbox_min))
    return target, max(half_extent, 1e-6), max(depth_extent, 1e-6)


def letterbox(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def info_tile(size: int, sample_id: str) -> np.ndarray:
    tile = np.full((size, size, 3), 250, dtype=np.uint8)
    lines = [
        f"ID: {sample_id}",
        "Shared normal + DINO features",
        "Shared post-SS RNG state",
        "Only SS structure differs",
        "Left: input   Right: prediction",
    ]
    for index, line in enumerate(lines):
        cv2.putText(tile, line, (34, 80 + index * 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    return tile


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = RenderOptions(
        resolution=args.resolution,
        padding=1.03,
        margin_pixels=18,
        background_color=(1.0, 1.0, 1.0, 1.0),
        mesh_color=(0.72, 0.72, 0.72, 1.0),
        label_bar_height=58,
        tile_gap=8,
    )

    merged_source = load_mesh(o3d, args.merged_source)
    merged_output = load_mesh(o3d, args.merged_output)
    merged_center, merged_half, merged_depth = shared_frame([merged_source, merged_output], options)
    merged_tiles = [
        render_mesh(o3d, item, FRONT_VIEW, merged_center, merged_half, merged_half, merged_depth, options)
        for item in (merged_source, merged_output)
    ]

    model_source = load_mesh(o3d, args.model_source)
    model_output = load_mesh(o3d, args.model_output)
    model_center, model_half, model_depth = shared_frame([model_source, model_output], options)
    model_tiles = [
        render_mesh(o3d, model_source, FRONT_VIEW, model_center, model_half, model_half, model_depth, options),
        render_mesh(o3d, model_output, FRONT_VIEW, model_center, model_half, model_half, model_depth, options),
    ]

    rows = [
        [draw_label(letterbox(args.normal, args.resolution), "Input normal", options.label_bar_height),
         draw_label(info_tile(args.resolution, args.sample_id), "Experiment", options.label_bar_height)],
        [draw_label(merged_tiles[0], args.row2_input_label, options.label_bar_height),
         draw_label(merged_tiles[1], args.row2_output_label, options.label_bar_height)],
        [draw_label(model_tiles[0], args.row3_input_label, options.label_bar_height),
         draw_label(model_tiles[1], args.row3_output_label, options.label_bar_height)],
    ]
    gap = np.full((options.label_bar_height + args.resolution, options.tile_gap, 3), 240, dtype=np.uint8)
    horizontal = [np.concatenate([row[0], gap, row[1]], axis=1) for row in rows]
    row_gap = np.full((options.tile_gap, horizontal[0].shape[1], 3), 240, dtype=np.uint8)
    montage = np.concatenate([horizontal[0], row_gap, horizontal[1], row_gap, horizontal[2]], axis=0)

    output_path = args.output_dir / "comparison_3x2.png"
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Failed to write {output_path}")
    cv2.imwrite(str(args.output_dir / "merged_input.png"), merged_tiles[0])
    cv2.imwrite(str(args.output_dir / "merged_output.png"), merged_tiles[1])
    cv2.imwrite(str(args.output_dir / "model_input_mesh.png"), model_tiles[0])
    cv2.imwrite(str(args.output_dir / "model_output.png"), model_tiles[1])
    (args.output_dir / "render_manifest.json").write_text(
        json.dumps({key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)}, indent=2),
        encoding="utf-8",
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
