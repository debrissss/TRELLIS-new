#!/usr/bin/env python3
"""Geometry metrics for mesh decoder evaluation."""

# 中文说明：
# mesh 几何指标公共工具模块，不作为独立命令行入口使用。
# 负责读取 mesh、采样表面点、计算 Chamfer、normal consistency、precision/recall/F-score。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh


@dataclass(frozen=True)
class MeshSamples:
    points: np.ndarray
    normals: np.ndarray


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError(f"Mesh contains non-finite vertices: {path}")
    return mesh


def mesh_basic_stats(mesh: trimesh.Trimesh, prefix: str) -> dict[str, float | int | bool]:
    extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    return {
        f"{prefix}_num_vertices": int(mesh.vertices.shape[0]),
        f"{prefix}_num_faces": int(mesh.faces.shape[0]),
        f"{prefix}_surface_area": float(mesh.area),
        f"{prefix}_bbox_extent_x": float(extents[0]),
        f"{prefix}_bbox_extent_y": float(extents[1]),
        f"{prefix}_bbox_extent_z": float(extents[2]),
        f"{prefix}_bbox_diag": float(np.linalg.norm(extents)),
        f"{prefix}_is_watertight": bool(mesh.is_watertight),
        f"{prefix}_components": int(len(mesh.split(only_watertight=False))),
    }


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> MeshSamples:
    if count <= 0:
        raise ValueError(f"point_samples must be positive, got {count}")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_index = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_index], dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return MeshSamples(points=points, normals=normals)


def _nearest(src: MeshSamples, dst: MeshSamples) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(dst.points)
    dist, idx = tree.query(src.points, k=1, workers=-1)
    return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def compare_meshes(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    *,
    point_samples: int,
    seed: int,
    fscore_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict[str, Any]:
    """Compare prediction and GT meshes with fixed random surface sampling."""
    pred = sample_surface(pred_mesh, point_samples, seed)
    gt = sample_surface(gt_mesh, point_samples, seed + 1)

    pred_to_gt, pred_nn = _nearest(pred, gt)
    gt_to_pred, gt_nn = _nearest(gt, pred)

    metrics: dict[str, Any] = {}
    metrics.update(mesh_basic_stats(pred_mesh, "pred"))
    metrics.update(mesh_basic_stats(gt_mesh, "gt"))
    metrics["chamfer_l1"] = float(pred_to_gt.mean() + gt_to_pred.mean())
    metrics["chamfer_l2"] = float((pred_to_gt**2).mean() + (gt_to_pred**2).mean())
    metrics["pred_to_gt_mean"] = float(pred_to_gt.mean())
    metrics["gt_to_pred_mean"] = float(gt_to_pred.mean())
    metrics["pred_to_gt_p95"] = float(np.percentile(pred_to_gt, 95))
    metrics["gt_to_pred_p95"] = float(np.percentile(gt_to_pred, 95))

    pred_normals = pred.normals
    gt_normals = gt.normals
    pred_match_normals = gt_normals[pred_nn]
    gt_match_normals = pred_normals[gt_nn]
    pred_dot = np.abs(np.sum(pred_normals * pred_match_normals, axis=1))
    gt_dot = np.abs(np.sum(gt_normals * gt_match_normals, axis=1))
    metrics["normal_consistency"] = float((pred_dot.mean() + gt_dot.mean()) * 0.5)

    for threshold in fscore_thresholds:
        precision = float((pred_to_gt < threshold).mean())
        recall = float((gt_to_pred < threshold).mean())
        if precision + recall == 0:
            fscore = 0.0
        else:
            fscore = 2.0 * precision * recall / (precision + recall)
        suffix = str(threshold).replace(".", "p")
        metrics[f"precision_{suffix}"] = precision
        metrics[f"recall_{suffix}"] = recall
        metrics[f"fscore_{suffix}"] = float(fscore)

    return metrics
