from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.impl.slat_flow_gs_image_evaluation_impl import (
    compare_generation_runs,
    compute_image_pair_metrics,
)
from eval.common.impl.slat_flow_mesh_generation_impl import generate_flow_meshes
from eval.common.impl.slat_flow_mesh_evaluation_impl import compare_mesh_runs_to_gt
from eval.latent_distribution import compute_single_latent_stats, summarize_latent_rows


def test_summarize_latent_files_reports_distribution_and_finite_rate(tmp_path: Path) -> None:
    latent_dir = tmp_path / "latents" / "model"
    latent_dir.mkdir(parents=True)
    np.savez(latent_dir / "a.npz", feats=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), coords=np.zeros((2, 3), dtype=np.uint8))
    np.savez(latent_dir / "b.npz", feats=np.array([[5.0, 6.0]], dtype=np.float32), coords=np.ones((1, 3), dtype=np.uint8))

    paths = [latent_dir / "a.npz", latent_dir / "b.npz"]
    records = [compute_single_latent_stats(path) for path in paths]
    rows = [record[0] for record in records]
    summary = summarize_latent_rows(paths, rows, [], [record[1] for record in records])

    assert summary["num_files"] == 2
    assert summary["failed_count"] == 0
    assert summary["token_count"]["sum"] == 3
    assert summary["feats"]["finite_rate"] == 1.0
    assert summary["feats"]["mean"] == 3.5
    assert [row["token_count"] for row in rows] == [2, 1]


def test_compute_image_pair_metrics_handles_mask_iou(tmp_path: Path) -> None:
    gt = Image.new("RGB", (4, 4), "black")
    pred = Image.new("RGB", (4, 4), "black")
    gt_pixels = gt.load()
    pred_pixels = pred.load()
    for x in range(2):
        for y in range(2):
            gt_pixels[x, y] = (255, 255, 255)
    for x in range(1, 3):
        for y in range(2):
            pred_pixels[x, y] = (255, 255, 255)

    metrics = compute_image_pair_metrics(pred, gt, skip_lpips=True)

    assert metrics["failed"] is False
    assert metrics["mask_iou"] == 1 / 3
    assert metrics["l1"] > 0
    assert metrics["psnr"] < 100


def test_compare_generation_runs_writes_run_level_summary(tmp_path: Path) -> None:
    for run_name, color in {"base": 32, "fine": 64}.items():
        sample_dir = tmp_path / run_name / "samples" / "sha0"
        sample_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4), (color, color, color)).save(sample_dir / "generated_grid.png")
        Image.new("RGB", (4, 4), (64, 64, 64)).save(sample_dir / "gt_grid.png")
        (tmp_path / run_name / "manifest.csv").write_text("sha256,index\nsha0,0\n", encoding="utf-8")

    output_dir = tmp_path / "compare"
    summary = compare_generation_runs({"base": tmp_path / "base", "fine": tmp_path / "fine"}, output_dir, skip_lpips=True)

    assert summary["runs"]["base"]["num_records"] == 1
    assert summary["runs"]["fine"]["num_records"] == 1
    assert summary["runs"]["fine"]["metrics"]["l1"]["mean"] == 0.0
    assert (output_dir / "comparison.csv").is_file()
    assert json.loads((output_dir / "summary.json").read_text())["failed_count"] == 0


def test_mesh_evaluation_consumes_existing_mesh_manifest(tmp_path: Path) -> None:
    sample_id = "sha0"
    mesh_run = tmp_path / "mesh_generation" / "run0"
    pred_path = mesh_run / "meshes" / f"{sample_id}.ply"
    gt_path = tmp_path / "dataset" / "renders" / sample_id / "mesh.ply"
    pred_path.parent.mkdir(parents=True)
    gt_path.parent.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.export(pred_path)
    mesh.export(gt_path)
    (mesh_run / "manifest.csv").write_text(
        "sample_id,mesh_path,failed,error\n"
        f"{sample_id},{pred_path},False,\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "mesh_metrics"
    summary = compare_mesh_runs_to_gt(
        runs={"run0": mesh_run},
        data_dir=tmp_path / "dataset",
        output_dir=output_dir,
        point_samples=200,
        seed=7,
        require_all_samples=True,
    )

    assert summary["failed_count"] == 0
    assert summary["runs"]["run0"]["successful_samples"] == 1
    assert (output_dir / "run0_per_sample.csv").is_file()
    assert not (output_dir / "meshes").exists()


def test_mesh_generation_writes_artifacts_without_metrics(tmp_path: Path) -> None:
    sample_id = "sha0"
    flow_run = tmp_path / "flow_run"
    latent_path = flow_run / "samples" / sample_id / "generated_latent.npz"
    latent_path.parent.mkdir(parents=True)
    np.savez(
        latent_path,
        coords=np.array([[1, 2, 3]], dtype=np.uint8),
        feats=np.zeros((1, 8), dtype=np.float32),
    )
    flow_config = tmp_path / "flow_config.json"
    flow_config.write_text('{"dataset": {"args": {}}}', encoding="utf-8")
    (flow_run / "summary.json").write_text(
        json.dumps({"config": str(flow_config)}),
        encoding="utf-8",
    )
    (flow_run / "manifest.csv").write_text(
        "sample_id,generated_latent_path,failed,error\n"
        f"{sample_id},{latent_path},False,\n",
        encoding="utf-8",
    )
    mesh_config = tmp_path / "mesh_config.json"
    mesh_config.write_text('{"models": {"decoder": {}}}', encoding="utf-8")

    def fake_decode(_decoder, _coords, _feats, output_path, _device):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)
        return mesh

    output_dir = tmp_path / "mesh_generation"
    module = "eval.common.impl.slat_flow_mesh_generation_impl"
    with (
        patch(f"{module}.build_stable3dgen_mesh_decoder", return_value=object()),
        patch(f"{module}.load_decoder_checkpoint"),
        patch(f"{module}.decode_latent_arrays_to_mesh", side_effect=fake_decode),
    ):
        summary = generate_flow_meshes(
            runs={"run0": flow_run},
            mesh_config_path=mesh_config,
            mesh_decoder_ckpt=tmp_path / "decoder.pt",
            run_mesh_decoders=None,
            output_dir=output_dir,
            device_name="cpu",
            require_all_samples=True,
        )

    run_output = output_dir / "run0"
    assert summary["failed_count"] == 0
    assert (run_output / "meshes" / f"{sample_id}.ply").is_file()
    assert (run_output / "manifest.csv").is_file()
    assert (run_output / "summary.json").is_file()
    assert not (run_output / "per_sample.csv").exists()


def _run_without_pytest() -> None:
    import tempfile

    tests = [
        test_summarize_latent_files_reports_distribution_and_finite_rate,
        test_compute_image_pair_metrics_handles_mask_iou,
        test_compare_generation_runs_writes_run_level_summary,
        test_mesh_evaluation_consumes_existing_mesh_manifest,
        test_mesh_generation_writes_artifacts_without_metrics,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("[OK] eval tool tests passed")


if __name__ == "__main__":
    _run_without_pytest()
