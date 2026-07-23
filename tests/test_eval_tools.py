from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.latent_stats import summarize_latent_files
from eval.flow_generation_metrics import compare_generation_runs, compute_image_pair_metrics


def test_summarize_latent_files_reports_distribution_and_finite_rate(tmp_path: Path) -> None:
    latent_dir = tmp_path / "latents" / "model"
    latent_dir.mkdir(parents=True)
    np.savez(latent_dir / "a.npz", feats=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), coords=np.zeros((2, 3), dtype=np.uint8))
    np.savez(latent_dir / "b.npz", feats=np.array([[5.0, 6.0]], dtype=np.float32), coords=np.ones((1, 3), dtype=np.uint8))

    summary, rows = summarize_latent_files([latent_dir / "a.npz", latent_dir / "b.npz"])

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


def _run_without_pytest() -> None:
    import tempfile

    tests = [
        test_summarize_latent_files_reports_distribution_and_finite_rate,
        test_compute_image_pair_metrics_handles_mask_iou,
        test_compare_generation_runs_writes_run_level_summary,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("[OK] eval tool tests passed")


if __name__ == "__main__":
    _run_without_pytest()
