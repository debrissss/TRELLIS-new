"""评估已生成的 SLat Flow 三角网格，不加载 flow 或 mesh decoder。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.common.dataset import render_mesh_path
from eval.common.io import safe_tag, write_csv, write_json
from eval.common.mesh_metrics import compare_meshes, load_mesh
from eval.common.slat_flow import resolve_repo_path


def load_mesh_generation_manifest(run_dir: Path) -> list[dict[str, str]]:
    """读取 mesh 生成清单；纯 mesh 目录则按文件名构造清单。"""
    manifest_path = run_dir / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"Mesh generation manifest is empty: {manifest_path}")
        return rows

    mesh_dir = run_dir / "meshes"
    if not mesh_dir.is_dir():
        mesh_dir = run_dir
    mesh_paths = sorted(mesh_dir.glob("*.ply"))
    if not mesh_paths:
        raise FileNotFoundError(
            f"No manifest.csv or *.ply meshes found in mesh run: {run_dir}"
        )
    return [
        {
            "sample_id": path.stem,
            "mesh_path": str(path),
            "failed": "False",
            "error": "",
        }
        for path in mesh_paths
    ]


def resolve_predicted_mesh_path(
    mesh_run_dir: Path,
    manifest_row: dict[str, str],
) -> Path:
    """定位 mesh 生成清单中的预测网格。"""
    mesh_text = manifest_row.get("mesh_path", "").strip()
    if mesh_text:
        return resolve_repo_path(Path(mesh_text))
    return mesh_run_dir / "meshes" / f"{manifest_row['sample_id']}.ply"


def summarize_rows(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总 mesh 指标的均值、中位数和标准差。"""
    successful = [row for row in rows if not row.get("failed")]
    summary: dict[str, Any] = {
        "num_records": len(rows),
        "successful_samples": len(successful),
        "failed_count": len(failures),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
    }
    numeric_keys = sorted(
        {
            key
            for row in successful
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and key != "sample_index"
            and math.isfinite(float(value))
        }
    )
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in successful
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(array.mean())
            summary[f"{key}_median"] = float(np.median(array))
            summary[f"{key}_std"] = float(array.std(ddof=0))
    return summary


def evaluate_single_mesh_run(
    *,
    name: str,
    mesh_run_dir: Path,
    data_dir: Path,
    point_samples: int,
    seed: int,
    require_all_samples: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """比较一组已经存在的预测 mesh 与 GT mesh。"""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_index, mesh_row in enumerate(
        load_mesh_generation_manifest(mesh_run_dir)
    ):
        sample_id = mesh_row["sample_id"]
        pred_mesh_path = resolve_predicted_mesh_path(mesh_run_dir, mesh_row)
        gt_mesh_path = render_mesh_path(data_dir, sample_id)
        row: dict[str, Any] = {
            "run": name,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "pred_mesh_path": str(pred_mesh_path),
            "gt_mesh_path": str(gt_mesh_path),
            "failed": False,
            "error": "",
        }
        try:
            if str(mesh_row.get("failed", "")).lower() in {"1", "true", "yes"}:
                raise RuntimeError(
                    f"Mesh generation failed for {sample_id}: "
                    f"{mesh_row.get('error', '')}"
                )
            pred_mesh = load_mesh(pred_mesh_path)
            gt_mesh = load_mesh(gt_mesh_path)
            row.update(
                compare_meshes(
                    pred_mesh,
                    gt_mesh,
                    point_samples=point_samples,
                    seed=seed + sample_index * 1009,
                )
            )
            print(
                f"[mesh-evaluation][{name}][{sample_index + 1}] OK {sample_id}",
                flush=True,
            )
        except Exception as exc:
            row.update({"failed": True, "error": repr(exc)})
            failures.append(row)
            print(
                f"[mesh-evaluation][{name}][{sample_index + 1}] "
                f"FAIL {sample_id}: {exc!r}",
                flush=True,
            )
            if require_all_samples:
                raise
        rows.append(row)

    summary = summarize_rows(rows, failures)
    summary.update(
        {
            "run": name,
            "mesh_run_dir": str(mesh_run_dir),
            "data_dir": str(data_dir),
            "point_samples": point_samples,
            "seed": seed,
        }
    )
    return summary, rows


def compare_mesh_runs_to_gt(
    *,
    runs: dict[str, Path],
    data_dir: Path,
    output_dir: Path,
    point_samples: int,
    seed: int,
    require_all_samples: bool = False,
) -> dict[str, Any]:
    """批量评估多个 mesh 生成任务，整个过程不加载任何神经网络。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for name, mesh_run_dir in runs.items():
        summary, rows = evaluate_single_mesh_run(
            name=name,
            mesh_run_dir=mesh_run_dir,
            data_dir=data_dir,
            point_samples=point_samples,
            seed=seed,
            require_all_samples=require_all_samples,
        )
        summaries[name] = summary
        all_rows.extend(rows)
        write_csv(output_dir / f"{safe_tag(name)}_per_sample.csv", rows)
        write_json(output_dir / f"{safe_tag(name)}_summary.json", summary)

    comparison = {
        "runs": summaries,
        "num_records": len(all_rows),
        "failed_count": sum(1 for row in all_rows if row.get("failed")),
        "data_dir": str(data_dir),
        "point_samples": point_samples,
        "seed": seed,
    }
    write_csv(output_dir / "per_sample.csv", all_rows)
    write_csv(output_dir / "all_runs_summary.csv", list(summaries.values()))
    write_json(output_dir / "summary.json", comparison)
    return comparison
