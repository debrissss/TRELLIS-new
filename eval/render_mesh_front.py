#!/usr/bin/env python3
"""Render one normalized mesh from the front with a fitted orthographic camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--padding", type=float, default=1.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.margin < 0 or args.margin * 2 >= args.resolution:
        raise ValueError("--margin must be in [0, resolution / 2)")
    if args.padding <= 0:
        raise ValueError("--padding must be positive")

    mesh = o3d.io.read_triangle_mesh(
        str(args.mesh.expanduser().resolve()), enable_post_processing=True
    )
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"Empty or invalid triangle mesh: {args.mesh}")
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    # TRELLIS/FaceScan normalized coordinates use +Z as the frontal eye axis.
    eye_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = -eye_axis
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    projected = np.stack(
        ((vertices - center) @ right, (vertices - center) @ up), axis=1
    )
    projected_min = projected.min(axis=0)
    projected_max = projected.max(axis=0)
    projected_center = (projected_min + projected_max) * 0.5
    target = center + right * projected_center[0] + up * projected_center[1]
    visible_pixels = args.resolution - 2 * args.margin
    half_extent = (
        float(np.max(projected_max - projected_min))
        * 0.5
        * args.resolution
        / visible_pixels
        * args.padding
    )
    depth_extent = max(float(np.max(bbox_max - bbox_min)), half_extent * 2, 1e-6)

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.resolution, args.resolution
    )
    renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = (0.72, 0.72, 0.72, 1.0)
    if hasattr(material, "roughness"):
        material.roughness = 0.65
    renderer.scene.add_geometry("mesh", mesh, material)
    try:
        profile = o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS
        renderer.scene.set_lighting(
            profile, np.array([0.0, -1.0, -1.0], dtype=np.float32)
        )
    except Exception:
        pass

    distance = depth_extent * 2.5
    eye = target + eye_axis * distance
    near = max(depth_extent * 0.01, 1e-4)
    far = max(depth_extent * 10.0, distance + depth_extent * 4.0)
    camera = renderer.scene.camera
    camera.look_at(target, eye, up)
    projection = o3d.visualization.rendering.Camera.Projection.Ortho
    camera.set_projection(
        projection,
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        near,
        far,
    )

    rgba = np.asarray(renderer.render_to_image())
    image = cv2.cvtColor(
        rgba, cv2.COLOR_RGBA2BGR if rgba.shape[2] == 4 else cv2.COLOR_RGB2BGR
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Failed to write {args.output}")
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
