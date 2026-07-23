import math
import tempfile
import unittest
from pathlib import Path

import torch

from eval.evaluate_ss_enc_dec_reconstruction import (
    compute_binary_metrics,
    compute_reconstruction_metrics,
    evaluate_checkpoint,
    summarize_metric_rows,
    write_occupied_points_ply,
)


class SparseStructureEvalMetricTests(unittest.TestCase):
    def test_compute_binary_metrics_reports_overlap_and_volume_bias(self):
        gt = torch.tensor([[[True, True, False, False]]])
        pred = torch.tensor([[[True, False, True, False]]])

        metrics = compute_binary_metrics(pred, gt)

        self.assertEqual(metrics["gt_occupied_voxels"], 2)
        self.assertEqual(metrics["predicted_occupied_voxels"], 2)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["dice_f1"], 0.5)
        self.assertAlmostEqual(metrics["occupancy_ratio"], 1.0)

    def test_compute_binary_metrics_marks_empty_gt_ratio_nan(self):
        gt = torch.zeros((1, 1, 4), dtype=torch.bool)
        pred = torch.zeros((1, 1, 4), dtype=torch.bool)

        metrics = compute_binary_metrics(pred, gt)

        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice_f1"], 1.0)
        self.assertTrue(math.isnan(metrics["occupancy_ratio"]))

    def test_compute_reconstruction_metrics_includes_soft_dice_loss(self):
        gt = torch.tensor([[[1.0, 0.0]]])
        logits = torch.tensor([[[0.0, 0.0]]])

        metrics = compute_reconstruction_metrics(logits, gt)

        self.assertAlmostEqual(metrics["soft_dice_loss"], 1 / 3)
        self.assertEqual(metrics["gt_occupied_voxels"], 1)
        self.assertEqual(metrics["predicted_occupied_voxels"], 0)

    def test_summarize_metric_rows_ignores_nan_values(self):
        rows = [
            {
                "iou": 1.0,
                "dice_f1": 1.0,
                "occupancy_ratio": 1.0,
                "soft_dice_loss": 0.1,
                "gt_occupied_voxels": 10,
                "predicted_occupied_voxels": 10,
            },
            {
                "iou": 0.0,
                "dice_f1": 0.5,
                "occupancy_ratio": float("nan"),
                "soft_dice_loss": 0.3,
                "gt_occupied_voxels": 0,
                "predicted_occupied_voxels": 0,
            },
        ]

        summary = summarize_metric_rows(rows)

        self.assertEqual(summary["num_samples"], 2)
        self.assertAlmostEqual(summary["iou"]["mean"], 0.5)
        self.assertAlmostEqual(summary["dice_f1"]["mean"], 0.75)
        self.assertAlmostEqual(summary["occupancy_ratio"]["mean"], 1.0)
        self.assertAlmostEqual(summary["soft_dice_loss"]["mean"], 0.2)

    def test_evaluate_checkpoint_can_request_posterior_sampling(self):
        class Dataset:
            instances = [("/tmp/root", "sha-a")]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"ss": torch.ones((1, 1, 1, 1), dtype=torch.long)}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_posterior_values = []

            def forward(self, x, sample_posterior=False):
                self.sample_posterior_values.append(sample_posterior)
                return x

        class Decoder(torch.nn.Module):
            def forward(self, z):
                return torch.ones_like(z)

        encoder = Encoder()
        rows = evaluate_checkpoint(
            checkpoint_name="dummy",
            model_pair={"encoder": encoder, "decoder": Decoder()},
            dataset=Dataset(),
            batch_size=1,
            device=torch.device("cpu"),
            sample_posterior=True,
        )

        self.assertEqual(encoder.sample_posterior_values, [True])
        self.assertEqual(rows[0]["sha256"], "sha-a")

    def test_write_occupied_points_ply_exports_voxel_centers(self):
        occupancy = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
        occupancy[0, 0, 0, 0] = True
        occupancy[0, 1, 1, 1] = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "occupancy.ply"
            write_occupied_points_ply(path, occupancy)
            content = path.read_text(encoding="utf-8")

        self.assertIn("element vertex 2", content)
        self.assertIn("-0.25000000 -0.25000000 -0.25000000", content)
        self.assertIn("0.25000000 0.25000000 0.25000000", content)


if __name__ == "__main__":
    unittest.main()
