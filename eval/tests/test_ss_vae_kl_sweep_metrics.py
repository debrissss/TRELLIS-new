import math
import unittest

import torch

from eval.evaluate_ss_vae_kl_sweep import (
    compute_latent_statistics,
    compute_reconstruction_metrics,
    compute_structural_metrics,
    compute_surface_metrics,
    pairwise_dice,
    sliced_wasserstein_to_standard_normal,
)


class SparseStructureVaeKlSweepMetricTests(unittest.TestCase):
    def test_reconstruction_metrics_report_fp_fn_and_stable_bce(self):
        gt = torch.tensor([[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]], dtype=torch.bool)
        logits = torch.tensor([[[[2.0, 1.0], [-2.0, -2.0]], [[-2.0, -2.0], [-2.0, -2.0]]]])

        metrics = compute_reconstruction_metrics(logits, gt)

        self.assertEqual(metrics["false_positive_voxels"], 1)
        self.assertEqual(metrics["false_negative_voxels"], 0)
        self.assertEqual(metrics["error_voxels"], 1)
        self.assertTrue(math.isfinite(metrics["bce_with_logits"]))

    def test_surface_metrics_are_zero_for_identical_occupancies(self):
        occupancy = torch.zeros((1, 4, 4, 4), dtype=torch.bool)
        occupancy[0, 1:3, 1:3, 1:3] = True

        metrics = compute_surface_metrics(occupancy, occupancy.clone())

        self.assertEqual(metrics["chamfer_distance"], 0.0)
        self.assertEqual(metrics["average_surface_distance"], 0.0)
        self.assertEqual(metrics["hd95"], 0.0)

    def test_pairwise_dice_detects_draw_disagreement(self):
        predictions = torch.tensor([
            [[[True, False]]],
            [[[True, True]]],
        ])

        self.assertAlmostEqual(pairwise_dice(predictions), 2 / 3)

    def test_latent_statistics_match_standard_normal_posterior(self):
        means = torch.zeros((3, 2, 2, 1, 1))
        logvars = torch.zeros_like(means)

        overall, sample_rows, channel_rows = compute_latent_statistics(
            means, logvars, active_threshold=1e-2, collapse_threshold=1e-3
        )

        self.assertEqual(overall["raw_kl_mean"], 0.0)
        self.assertEqual(overall["aggregate_mean_abs_deviation"], 0.0)
        self.assertEqual(overall["aggregate_variance_abs_deviation"], 0.0)
        self.assertEqual(overall["active_ratio"], 0.0)
        self.assertEqual(overall["collapse_ratio"], 1.0)
        self.assertEqual(len(sample_rows), 3)
        self.assertEqual(len(channel_rows), 2)

    def test_structural_metrics_report_components(self):
        occupancy = torch.zeros((1, 5, 5, 5), dtype=torch.bool)
        occupancy[0, 0, 0, 0] = True
        occupancy[0, 4, 4, 4] = True

        metrics = compute_structural_metrics(occupancy, dense_threshold=0.5)

        self.assertEqual(metrics["connected_components"], 2)
        self.assertAlmostEqual(metrics["largest_component_ratio"], 0.5)
        self.assertFalse(metrics["is_empty"])

    def test_sliced_wasserstein_is_finite_and_reproducible(self):
        means = torch.zeros((4, 2, 2, 1, 1))
        logvars = torch.zeros_like(means)

        first = sliced_wasserstein_to_standard_normal(means, logvars, 8, 100, seed=123)
        second = sliced_wasserstein_to_standard_normal(means, logvars, 8, 100, seed=123)

        self.assertTrue(math.isfinite(first))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
