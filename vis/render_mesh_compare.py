#!/usr/bin/env python3
"""Render multiview mesh comparison images for aligned eval results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


SHA256_LEN = 64
DEFAULT_CONFIG = Path(__file__).with_name("mesh_compare_config.json")


@dataclass(frozen=True)
class Source:
    label: str
    type: str
    path: Path
    mesh_name: str = "mesh.ply"


@dataclass(frozen=True)
class RenderOptions:
    resolution: int
    padding: float
    margin_pixels: int
    background_color: tuple[float, float, float, float]
    mesh_color: tuple[float, float, float, float]
    label_bar_height: int
    tile_gap: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render front/left/right mesh comparison strips for eval results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument(
        "--padding",
        type=float,
        default=None,
        help="Extra orthographic scale multiplier after margin fitting. Smaller values zoom in; default comes from config.",
    )
    parser.add_argument(
        "--margin-pixels",
        type=int,
        default=None,
        help="Target minimum image margin around the rendered model. Default comes from config.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--sha256",
        action="append",
        default=None,
        help="Render only this sha256. Can be passed more than once.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "sources" not in config or not isinstance(config["sources"], list):
        raise ValueError("Config must contain a list field: sources")
    if not config["sources"]:
        raise ValueError("Config sources cannot be empty")
    if "views" not in config or not isinstance(config["views"], dict):
        raise ValueError("Config must contain a dict field: views")
    return config


def parse_sources(config: dict[str, Any]) -> list[Source]:
    sources: list[Source] = []
    labels: set[str] = set()
    for idx, item in enumerate(config["sources"]):
        label = str(item.get("label", "")).strip()
        source_type = str(item.get("type", "mesh_dir")).strip()
        path = Path(str(item.get("path", ""))).expanduser()
        mesh_name = str(item.get("mesh_name", "mesh.ply"))
        if not label:
            raise ValueError(f"sources[{idx}] is missing label")
        if label in labels:
            raise ValueError(f"Duplicate source label: {label}")
        if not path:
            raise ValueError(f"sources[{idx}] is missing path")
        if source_type not in {"mesh_dir", "gt_facescape"}:
            raise ValueError(f"Unsupported source type for {label}: {source_type}")
        labels.add(label)
        sources.append(Source(label=label, type=source_type, path=path, mesh_name=mesh_name))
    return sources


def parse_render_options(
    config: dict[str, Any],
    resolution_override: int | None,
    padding_override: float | None,
    margin_pixels_override: int | None,
) -> RenderOptions:
    render = config.get("render", {})
    resolution = int(resolution_override or render.get("resolution", 1024))
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    padding = float(padding_override if padding_override is not None else render.get("padding", 1.0))
    if padding <= 0:
        raise ValueError("padding must be positive")
    margin_pixels = int(
        margin_pixels_override if margin_pixels_override is not None else render.get("margin_pixels", 10)
    )
    if margin_pixels < 0:
        raise ValueError("margin-pixels must be non-negative")
    if margin_pixels * 2 >= resolution:
        raise ValueError("margin-pixels must be less than half of resolution")
    background_color = tuple(render.get("background_color", [1.0, 1.0, 1.0, 1.0]))
    mesh_color = tuple(render.get("mesh_color", [0.78, 0.78, 0.78, 1.0]))
    if len(background_color) != 4:
        raise ValueError("render.background_color must contain 4 values")
    if len(mesh_color) != 4:
        raise ValueError("render.mesh_color must contain 4 values")
    return RenderOptions(
        resolution=resolution,
        padding=padding,
        margin_pixels=margin_pixels,
        background_color=background_color,
        mesh_color=mesh_color,
        label_bar_height=int(render.get("label_bar_height", 72)),
        tile_gap=int(render.get("tile_gap", 8)),
    )


def validate_sha256(value: str) -> str:
    value = value.strip()
    if len(value) != SHA256_LEN or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Invalid sha256: {value}")
    return value.lower()


def index_source(source: Source) -> dict[str, Path]:
    if not source.path.is_dir():
        raise FileNotFoundError(f"Source path does not exist for {source.label}: {source.path}")

    indexed: dict[str, Path] = {}
    if source.type == "mesh_dir":
        candidates = source.path.glob("*.ply")
        for mesh_path in candidates:
            stem = mesh_path.stem.lower()
            if len(stem) == SHA256_LEN:
                indexed[stem] = mesh_path
    elif source.type == "gt_facescape":
        for item in source.path.iterdir():
            if not item.is_dir():
                continue
            sha = item.name.lower()
            if len(sha) != SHA256_LEN:
                continue
            mesh_path = item / source.mesh_name
            if mesh_path.is_file():
                indexed[sha] = mesh_path
    return indexed


def build_index(sources: list[Source]) -> dict[str, dict[str, Path]]:
    return {source.label: index_source(source) for source in sources}


def common_sha256s(index: dict[str, dict[str, Path]]) -> list[str]:
    sets = [set(paths) for paths in index.values()]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def filtered_sha256s(all_sha256s: list[str], requested: list[str] | None, max_items: int | None) -> list[str]:
    selected = all_sha256s
    if requested:
        wanted = {validate_sha256(item) for item in requested}
        missing = sorted(wanted - set(all_sha256s))
        if missing:
            print(f"Warning: requested sha256 values not present in every source: {', '.join(missing)}")
        selected = [sha for sha in all_sha256s if sha in wanted]
    if max_items is not None:
        if max_items <= 0:
            raise ValueError("max-items must be positive")
        selected = selected[:max_items]
    return selected


def print_summary(sources: list[Source], index: dict[str, dict[str, Path]], all_sha256s: list[str], selected: list[str]) -> None:
    print("Sources:")
    for source in sources:
        print(f"  {source.label}: {len(index[source.label])} meshes ({source.path})")
    print(f"Common sha256 count: {len(all_sha256s)}")
    print(f"Selected sha256 count: {len(selected)}")
    if selected:
        preview = ", ".join(selected[:5])
        suffix = " ..." if len(selected) > 5 else ""
        print(f"Selected preview: {preview}{suffix}")


def require_render_dependencies() -> Any:
    if np is None:
        raise RuntimeError(
            "Missing dependency: numpy. Run this script in the project environment "
            "that has NumPy installed, or install it before rendering."
        )
    if cv2 is None:
        raise RuntimeError(
            "Missing dependency: cv2/opencv-python. Run this script in the project environment "
            "that has OpenCV installed, or install it before rendering."
        )
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: open3d. Run this script in the project environment "
            "that has Open3D installed, or install it before rendering."
        ) from exc
    return o3d


def load_mesh(o3d: Any, path: Path) -> Any:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Empty or invalid mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def mesh_bbox(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    bbox = mesh.get_axis_aligned_bounding_box()
    return np.asarray(bbox.min_bound, dtype=np.float64), np.asarray(bbox.max_bound, dtype=np.float64)


def combined_bbox(meshes: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for mesh in meshes:
        mn, mx = mesh_bbox(mesh)
        mins.append(mn)
        maxs.append(mx)
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def view_frame(
    meshes: list[Any],
    center: np.ndarray,
    eye_axis: np.ndarray,
    up: np.ndarray,
    options: RenderOptions,
) -> tuple[np.ndarray, float, float]:
    forward = -eye_axis
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        raise ValueError("View up vector cannot be parallel to eye_axis")
    right = right / right_norm
    up = up / np.linalg.norm(up)

    projected_mins: list[np.ndarray] = []
    projected_maxs: list[np.ndarray] = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64) - center
        projected = np.stack((vertices @ right, vertices @ up), axis=1)
        projected_mins.append(np.min(projected, axis=0))
        projected_maxs.append(np.max(projected, axis=0))

    projected_min = np.min(np.stack(projected_mins, axis=0), axis=0)
    projected_max = np.max(np.stack(projected_maxs, axis=0), axis=0)
    projected_size = projected_max - projected_min
    projected_center = (projected_min + projected_max) * 0.5
    fit_size = float(np.max(projected_size))
    if fit_size <= 0:
        fit_size = 1e-6

    visible_pixels = options.resolution - options.margin_pixels * 2
    half_extent = fit_size * 0.5 * options.resolution / visible_pixels * options.padding
    target = center + right * projected_center[0] + up * projected_center[1]
    return target, half_extent, half_extent


def make_material(o3d: Any, color: tuple[float, float, float, float]) -> Any:
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = color
    if hasattr(material, "roughness"):
        material.roughness = 0.65
    return material


def setup_camera(
    o3d: Any,
    renderer: Any,
    center: np.ndarray,
    half_width: float,
    half_height: float,
    depth_extent: float,
    eye_axis: np.ndarray,
    up: np.ndarray,
) -> None:
    safe_extent = max(float(depth_extent), half_width * 2.0, half_height * 2.0, 1e-6)
    distance = safe_extent * 2.5
    eye = center + eye_axis * distance
    near = max(safe_extent * 0.01, 1e-4)
    far = max(safe_extent * 10.0, distance + safe_extent * 4.0)

    camera = renderer.scene.camera
    camera.look_at(center.astype(np.float64), eye.astype(np.float64), up.astype(np.float64))

    try:
        projection = o3d.visualization.rendering.Camera.Projection.Ortho
        camera.set_projection(projection, -half_width, half_width, -half_height, half_height, near, far)
    except Exception:
        renderer.setup_camera(35.0, center.astype(np.float64), eye.astype(np.float64), up.astype(np.float64))


def render_mesh(
    o3d: Any,
    mesh: Any,
    view: dict[str, Any],
    center: np.ndarray,
    half_width: float,
    half_height: float,
    depth_extent: float,
    options: RenderOptions,
) -> np.ndarray:
    renderer = o3d.visualization.rendering.OffscreenRenderer(options.resolution, options.resolution)
    renderer.scene.set_background(options.background_color)
    try:
        profile = o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS
        renderer.scene.set_lighting(profile, np.array([0.0, -1.0, -1.0], dtype=np.float32))
    except Exception:
        pass
    renderer.scene.add_geometry("mesh", mesh, make_material(o3d, options.mesh_color))

    eye_axis = np.asarray(view["eye_axis"], dtype=np.float64)
    eye_norm = np.linalg.norm(eye_axis)
    if eye_norm == 0:
        raise ValueError(f"Invalid zero eye_axis in view: {view}")
    eye_axis = eye_axis / eye_norm
    up = np.asarray(view.get("up", [0.0, 1.0, 0.0]), dtype=np.float64)
    setup_camera(o3d, renderer, center, half_width, half_height, depth_extent, eye_axis, up)

    image = np.asarray(renderer.render_to_image())
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_label(tile: np.ndarray, label: str, bar_height: int) -> np.ndarray:
    if bar_height <= 0:
        return tile
    h, w = tile.shape[:2]
    out = np.full((h + bar_height, w, 3), 255, dtype=np.uint8)
    out[bar_height:, :, :] = tile
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    while text_size[0] > w - 24 and scale > 0.35:
        scale *= 0.9
        text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    x = max((w - text_size[0]) // 2, 8)
    y = max((bar_height + text_size[1]) // 2, text_size[1] + 4)
    cv2.putText(out, label, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    cv2.line(out, (0, bar_height - 1), (w, bar_height - 1), (220, 220, 220), 1)
    return out


def stitch_tiles(tiles: list[np.ndarray], labels: list[str], options: RenderOptions) -> np.ndarray:
    labeled = [draw_label(tile, label, options.label_bar_height) for tile, label in zip(tiles, labels)]
    if options.tile_gap <= 0:
        return np.concatenate(labeled, axis=1)
    h = labeled[0].shape[0]
    gap = np.full((h, options.tile_gap, 3), 245, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for idx, tile in enumerate(labeled):
        if idx:
            parts.append(gap)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


def render_one_sha(
    o3d: Any,
    sha: str,
    sources: list[Source],
    index: dict[str, dict[str, Path]],
    views: dict[str, Any],
    output_dir: Path,
    options: RenderOptions,
) -> None:
    loaded: dict[str, Any] = {}
    for source in sources:
        loaded[source.label] = load_mesh(o3d, index[source.label][sha])

    bbox_min, bbox_max = combined_bbox(list(loaded.values()))
    center = (bbox_min + bbox_max) * 0.5
    depth_extent = float(np.max(bbox_max - bbox_min))
    labels = [source.label for source in sources]

    for view_name, view in views.items():
        eye_axis = np.asarray(view["eye_axis"], dtype=np.float64)
        eye_axis = eye_axis / np.linalg.norm(eye_axis)
        up = np.asarray(view.get("up", [0.0, 1.0, 0.0]), dtype=np.float64)
        target, half_width, half_height = view_frame(list(loaded.values()), center, eye_axis, up, options)
        tiles = [
            render_mesh(
                o3d,
                loaded[source.label],
                view,
                target,
                half_width,
                half_height,
                depth_extent,
                options,
            )
            for source in sources
        ]
        stitched = stitch_tiles(tiles, labels, options)
        output_path = output_dir / f"{sha}_{view_name}.png"
        if not cv2.imwrite(str(output_path), stitched):
            raise RuntimeError(f"Failed to write image: {output_path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sources = parse_sources(config)
    options = parse_render_options(config, args.resolution, args.padding, args.margin_pixels)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/vis/mesh_compare")).expanduser()

    index = build_index(sources)
    all_sha256s = common_sha256s(index)
    selected = filtered_sha256s(all_sha256s, args.sha256, args.max_items)
    print_summary(sources, index, all_sha256s, selected)

    if args.dry_run:
        print("Dry run only; no rendering performed.")
        return 0
    if not selected:
        print("No meshes selected; nothing to render.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    o3d = require_render_dependencies()
    failures: list[tuple[str, str]] = []
    for idx, sha in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] Rendering {sha}")
        try:
            render_one_sha(o3d, sha, sources, index, config["views"], output_dir, options)
        except Exception as exc:
            failures.append((sha, str(exc)))
            print(f"Warning: skipped {sha}: {exc}")

    print(f"Rendered sha256 count: {len(selected) - len(failures)}")
    print(f"Output directory: {output_dir}")
    if failures:
        print("Failures:")
        for sha, reason in failures:
            print(f"  {sha}: {reason}")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
