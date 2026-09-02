"""Rebuild FaceScan ``up_normal.png`` conditions from projected 3D support.

For every record in the prepared FaceScan ControlNet training split, the
un-normalized ``model.ply`` mesh is transformed from the registered model
frame back into the original up-view camera frame and projected with the up
color intrinsics.  Its triangles are rasterized directly to define a dense
target silhouette.  Pixels outside the result are removed; no new normal
pixels are synthesized.  The masked image is then cropped to a square around
the projection without resizing, so the original pixel scale is preserved.
The square crop is a translation only.  If it crosses the source canvas, the
missing area is padded with black rather than shifting or rescaling the
subject.  ``projection_crop.json`` records the crop and translated intrinsics.

Important coordinate convention
-------------------------------
``align_to_standard/up_T.txt`` maps the original ``model/up.ply`` into the
registered model frame in which ``model.ply`` was reconstructed.  Projection
into ``model/up_normal.png`` therefore uses:

    model.ply --inverse(up_T)--> up Color --K_color--> px

The points in ``model/up.ply`` already use the color-camera coordinate frame:
projecting them directly with ``Cointrinsics`` reproduces their source pixels.
Applying ``up.json/Extrinsics`` again would double-apply the IR-to-color
calibration and shift the projection horizontally by roughly 40 pixels.

The generated file atomically replaces the dataset symlink at
``renders_cond/<id>/up_normal.png``.  The original file under
``face_scan_test_data`` is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def read_ascii_triangle_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read xyz vertices and triangle indices from FaceScan ``model.ply``."""
    with path.open("r", encoding="ascii") as file:
        vertex_count = None
        face_count = None
        ply_format = None
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY end_header not found: {path}")
            text = line.strip()
            if text.startswith("format "):
                ply_format = text.split()[1]
            elif text.startswith("element vertex "):
                vertex_count = int(text.rsplit(" ", 1)[-1])
            elif text.startswith("element face "):
                face_count = int(text.rsplit(" ", 1)[-1])
            elif text == "end_header":
                break

        if ply_format != "ascii":
            raise ValueError(f"Expected ASCII model mesh, got {ply_format}: {path}")
        if vertex_count is None or face_count is None:
            raise ValueError(f"PLY vertex/face count not found: {path}")

        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            values = file.readline().split()
            if len(values) < 3:
                raise ValueError(f"Invalid vertex {index}/{vertex_count}: {path}")
            vertices[index] = [float(value) for value in values[:3]]

        faces = np.empty((face_count, 3), dtype=np.int32)
        for index in range(face_count):
            values = file.readline().split()
            if not values or int(values[0]) != 3 or len(values) < 4:
                raise ValueError(
                    f"Expected triangle at face {index}/{face_count}: {path}"
                )
            faces[index] = [int(value) for value in values[1:4]]
    return vertices, faces


def ensure_original_image_link(source: Path, destination: Path) -> None:
    """Keep an explicit original-image reference next to the processed PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(
        source.resolve(),
        destination.parent.resolve(),
    )
    if destination.is_symlink():
        if os.readlink(destination) == relative_target:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink original comparison image: "
            f"{destination}"
        )
    destination.symlink_to(relative_target)


def project_up_color_points(
    points: np.ndarray,
    color_intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project up color-camera points and return pixels plus depth."""
    z = points[:, 2]
    # FaceScan uses X-left, Y-up and Z-backward coordinates.  Z is negative in
    # front of the camera, so horizontal and vertical pixel formulas have
    # different explicit signs.  Using the conventional X-right formula here
    # mirrors the projected support and can retain only half of the real face.
    u = -color_intrinsics["fx"] * points[:, 0] / z
    u += color_intrinsics["ppx"]
    v = color_intrinsics["fy"] * points[:, 1] / z
    v += color_intrinsics["ppy"]
    return np.column_stack((u, v)), z


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are not connected to an image border."""
    flood = mask.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        raise ValueError("Projected mask contains no connected foreground")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def build_mesh_projection_mask(
    projected_vertices: np.ndarray,
    vertex_depths: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize projected model triangles into a dense silhouette mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    face_vertices = projected_vertices[faces]
    valid_faces = (
        np.isfinite(face_vertices).all(axis=(1, 2))
        & (vertex_depths[faces] < 0).all(axis=1)
    )
    if not valid_faces.any():
        raise ValueError("Projected model contains no visible triangles")
    triangles = np.rint(face_vertices[valid_faces]).astype(np.int32)
    cv2.fillPoly(mask, triangles, 255, lineType=cv2.LINE_8)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    return mask


def constrain_to_image_foreground(
    projection_mask: np.ndarray,
    image: np.ndarray,
    foreground_threshold: int,
) -> np.ndarray:
    """Keep real normal-map pixels supported by the projected target cloud."""
    if not 0 <= foreground_threshold <= 255:
        raise ValueError("foreground_threshold must be in [0, 255]")

    # Normal maps use black as invalid background.  Use the maximum channel so
    # valid normals are retained even if one or two encoded components are 0.
    foreground = np.where(
        image.max(axis=2) > foreground_threshold,
        255,
        0,
    ).astype(np.uint8)
    recovered = cv2.bitwise_and(projection_mask, foreground)

    # Retain the dominant target component after the projected support is
    # intersected with the actual normal-map foreground.
    return largest_component(recovered)


def apply_projection_mask(
    image: np.ndarray,
    mask: np.ndarray,
    feather_radius: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Remove invalid pixels while preserving the original camera canvas."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot apply an empty projection mask")
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1

    if feather_radius > 0:
        kernel = 2 * feather_radius + 1
        alpha = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=0)
    else:
        alpha = mask
    alpha = alpha.astype(np.float32) / 255.0
    masked = np.rint(image.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    return masked, (x_min, y_min, x_max, y_max)


def crop_square_without_resizing(
    image: np.ndarray,
    valid_bbox: Tuple[int, int, int, int],
    margin_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a projection-centered square using translation and padding only."""
    if margin_ratio < 0:
        raise ValueError("square_margin_ratio must be non-negative")
    x_min, y_min, x_max, y_max = valid_bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(f"Invalid projected bbox: {valid_bbox}")

    # margin_ratio is the margin on each side, measured relative to the larger
    # bbox dimension.  No resize is performed after this square is extracted.
    side = int(np.ceil(max(bbox_width, bbox_height) * (1 + 2 * margin_ratio)))
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    crop_x_min = int(np.floor(center_x - side / 2))
    crop_y_min = int(np.floor(center_y - side / 2))
    crop_x_max = crop_x_min + side
    crop_y_max = crop_y_min + side

    cropped = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
    source_height, source_width = image.shape[:2]
    source_x_min = max(crop_x_min, 0)
    source_y_min = max(crop_y_min, 0)
    source_x_max = min(crop_x_max, source_width)
    source_y_max = min(crop_y_max, source_height)
    if source_x_min < source_x_max and source_y_min < source_y_max:
        destination_x_min = source_x_min - crop_x_min
        destination_y_min = source_y_min - crop_y_min
        cropped[
            destination_y_min:destination_y_min + source_y_max - source_y_min,
            destination_x_min:destination_x_min + source_x_max - source_x_min,
        ] = image[source_y_min:source_y_max, source_x_min:source_x_max]
    return cropped, (crop_x_min, crop_y_min, crop_x_max, crop_y_max)


def process_instance(
    instance: str,
    source_dir: Path,
    output_path: Path,
    foreground_threshold: int,
    feather_radius: int,
    square_margin_ratio: float,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], float]:
    image_path = source_dir / "model" / "up_normal.png"
    camera_path = source_dir / "model" / "up.json"
    model_path = source_dir / "model.ply"
    up_transform_path = source_dir / "align_to_standard" / "up_T.txt"
    if not image_path.is_file():
        raise FileNotFoundError(f"up_normal.png missing for {instance}: {image_path}")
    if not camera_path.is_file():
        raise FileNotFoundError(f"up.json missing for {instance}: {camera_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model.ply missing for {instance}: {model_path}"
        )
    if not up_transform_path.is_file():
        raise FileNotFoundError(
            f"up_T.txt missing for {instance}: {up_transform_path}"
        )

    # 对比文件始终指向 face_scan_test_data 中的原图；处理后的 up_normal.png
    # 是独立真实文件，二者不会互相覆盖。
    ensure_original_image_link(
        image_path,
        output_path.with_name("up_normal_original.png"),
    )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with camera_path.open("r", encoding="utf-8") as file:
        camera = json.load(file)
    color_intrinsics = camera["Cointrinsics"]
    expected_shape = (
        int(color_intrinsics["height"]),
        int(color_intrinsics["width"]),
    )
    if image.shape[:2] != expected_shape:
        raise ValueError(
            f"Image/intrinsics size mismatch for {instance}: "
            f"image={image.shape[:2]}, color={expected_shape}"
        )

    vertices_aligned, faces = read_ascii_triangle_mesh(model_path)
    up_transform = np.loadtxt(up_transform_path, dtype=np.float64)
    if up_transform.shape != (4, 4) or not np.isfinite(up_transform).all():
        raise ValueError(
            f"Expected a finite 4x4 up transform for {instance}: "
            f"{up_transform_path}"
        )
    if not np.allclose(up_transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(
            f"Invalid homogeneous bottom row in {up_transform_path}: "
            f"{up_transform[3].tolist()}"
        )

    # up_T maps the original up-view color-camera points into the registered
    # frame in which model.ply was reconstructed, so undo it before applying
    # the original camera intrinsics.  Do not apply the JSON Extrinsics here:
    # model/up.ply already incorporates that calibration.
    inverse_up_transform = np.linalg.inv(up_transform)
    vertices_up_color = (
        vertices_aligned @ inverse_up_transform[:3, :3].T
        + inverse_up_transform[:3, 3]
    )
    pixels, z = project_up_color_points(vertices_up_color, color_intrinsics)
    finite = np.isfinite(pixels).all(axis=1) & (z < 0)
    pixels = pixels[finite]
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < expected_shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < expected_shape[0])
    )
    mask = build_mesh_projection_mask(
        pixels,
        z,
        faces,
        width=expected_shape[1],
        height=expected_shape[0],
    )
    mask = constrain_to_image_foreground(
        mask,
        image,
        foreground_threshold=foreground_threshold,
    )
    output, valid_bbox = apply_projection_mask(
        image,
        mask,
        feather_radius=feather_radius,
    )
    output, crop_box = crop_square_without_resizing(
        output,
        valid_bbox,
        margin_ratio=square_margin_ratio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # os.replace replaces the dataset symlink itself, never the original image
    # to which that symlink points.
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    encoded = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(temporary_path), encoded):
        raise OSError(f"Failed to write temporary image: {temporary_path}")
    os.replace(temporary_path, output_path)

    # Cropping changes only the image origin.  Focal lengths and pixel scale
    # stay unchanged; subtracting the crop origin gives the exact new principal
    # point, including when the crop extends outside the source canvas.
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box
    crop_metadata = {
        "source_image": str(image_path.resolve()),
        "source_size": [int(image.shape[1]), int(image.shape[0])],
        "valid_bbox_xyxy": list(valid_bbox),
        "crop_box_xyxy_in_source_coordinates": list(crop_box),
        "output_size": [int(output.shape[1]), int(output.shape[0])],
        "resized": False,
        "square_margin_ratio": square_margin_ratio,
        "intrinsics": {
            "fx": float(color_intrinsics["fx"]),
            "fy": float(color_intrinsics["fy"]),
            "ppx": float(color_intrinsics["ppx"] - crop_x_min),
            "ppy": float(color_intrinsics["ppy"] - crop_y_min),
            "width": int(output.shape[1]),
            "height": int(output.shape[0]),
        },
    }
    metadata_path = output_path.with_name("projection_crop.json")
    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.stem}.tmp.json"
    )
    with temporary_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(crop_metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary_metadata_path, metadata_path)
    return (
        valid_bbox,
        crop_box,
        float(in_frame.sum()) / max(len(vertices_aligned), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate projected FaceScan up-normal conditions"
    )
    parser.add_argument(
        "--dataset_root",
        default="datasets/FaceScan_ControlNet/train",
        help="Prepared FaceScan ControlNet training split.",
    )
    parser.add_argument(
        "--foreground_threshold",
        type=int,
        default=5,
        help="Maximum RGB value at or below this value is treated as background.",
    )
    parser.add_argument("--feather_radius", type=int, default=0)
    parser.add_argument(
        "--square_margin_ratio",
        type=float,
        default=0.08,
        help=(
            "Per-side margin relative to the larger projected bbox dimension; "
            "the resulting square is not resized."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open("r", newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    failures = []
    for record in records:
        instance = str(record["sha256"])
        source_dir = Path(record["local_path"])
        output_path = (
            dataset_root / "renders_cond" / instance / "up_normal.png"
        )
        try:
            valid_bbox, crop_box, in_frame_ratio = process_instance(
                instance,
                source_dir,
                output_path,
                foreground_threshold=args.foreground_threshold,
                feather_radius=args.feather_radius,
                square_margin_ratio=args.square_margin_ratio,
            )
            print(
                f"OK {instance}: valid_bbox={valid_bbox}, "
                f"crop_box={crop_box}, "
                f"in_frame={in_frame_ratio:.4%}, output={output_path}"
            )
        except Exception as error:
            failures.append((instance, str(error)))
            print(f"FAILED {instance}: {error}")

    if failures:
        details = "; ".join(
            f"{instance}: {reason}" for instance, reason in failures
        )
        raise RuntimeError(
            f"Failed to process {len(failures)}/{len(records)} samples: {details}"
        )
    print(f"Generated {len(records)} projected image conditions.")


if __name__ == "__main__":
    main()
