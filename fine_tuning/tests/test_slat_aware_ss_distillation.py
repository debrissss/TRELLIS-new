import unittest
from types import SimpleNamespace

import torch

from trellis.trainers.flow_matching.slat_aware_ss_distillation import (
    ImageConditionedSLatAwareSSFlowMatchingCFGTrainer_ControlNet,
    dense_occupancy_from_coords,
    flow_velocity_to_x0,
    sampled_distillation_weight,
    scheduled_distillation_weight,
    soft_dice_loss,
    sparse_candidate_logits,
)


class _FakeSparseTensor:
    def __init__(self, coords, feats):
        self.coords = coords
        self.feats = feats
        self.shape = torch.Size(
            [int(coords[:, 0].max()) + 1, feats.shape[1]]
        )

    @property
    def device(self):
        return self.feats.device

    def replace(self, feats):
        return _FakeSparseTensor(self.coords, feats)


class _FakeTeacher(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = torch.nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(channels))

    def forward(self, sparse_input, _t, _cond):
        return SimpleNamespace(feats=self.projection(sparse_input.feats))


class SLatAwareSSDistillationTest(unittest.TestCase):
    def test_flow_velocity_to_x0_matches_flow_parameterization(self):
        sigma_min = 1e-5
        x_0 = torch.randn(2, 3, 2, 2, 2)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0.2, 0.8])
        t_view = t.view(2, 1, 1, 1, 1)
        sigma_t = sigma_min + (1.0 - sigma_min) * t_view
        x_t = (1.0 - t_view) * x_0 + sigma_t * noise
        velocity = (1.0 - sigma_min) * noise - x_0

        recovered = flow_velocity_to_x0(x_t, velocity, t, sigma_min)

        torch.testing.assert_close(recovered, x_0)

    def test_candidate_gather_and_dense_target_use_same_coordinate_order(self):
        logits = torch.arange(
            2 * 4 * 4 * 4, dtype=torch.float32
        ).reshape(2, 1, 4, 4, 4)
        coords = torch.tensor(
            [[0, 1, 2, 3], [1, 3, 2, 1]], dtype=torch.int32
        )

        gathered = sparse_candidate_logits(logits, coords)
        target = dense_occupancy_from_coords(
            coords,
            batch_size=2,
            resolution=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            gathered,
            torch.stack(
                [logits[0, 0, 1, 2, 3], logits[1, 0, 3, 2, 1]]
            ),
        )
        self.assertEqual(target.sum().item(), 2)
        self.assertEqual(target[0, 0, 1, 2, 3].item(), 1)
        self.assertEqual(target[1, 0, 3, 2, 1].item(), 1)

    def test_soft_dice_prefers_correct_occupancy(self):
        target = torch.tensor([[[[[1.0, 0.0]]]]])
        correct = torch.tensor([[[[[10.0, -10.0]]]]])
        inverted = -correct

        self.assertLess(
            soft_dice_loss(correct, target),
            soft_dice_loss(inverted, target),
        )

    def test_distillation_schedule_has_start_and_linear_warmup(self):
        self.assertEqual(scheduled_distillation_weight(9, 10, 4), 0.0)
        self.assertEqual(scheduled_distillation_weight(10, 10, 4), 0.25)
        self.assertEqual(scheduled_distillation_weight(11, 10, 4), 0.5)
        self.assertEqual(scheduled_distillation_weight(13, 10, 4), 1.0)
        self.assertEqual(scheduled_distillation_weight(100, 10, 4), 1.0)

    def test_periodic_distillation_preserves_average_weight_by_default(self):
        self.assertEqual(
            sampled_distillation_weight(
                12, 10, 4, 4, preserve_average=True
            ),
            3.0,
        )
        self.assertEqual(
            sampled_distillation_weight(
                13, 10, 4, 4, preserve_average=True
            ),
            0.0,
        )
        self.assertEqual(
            sampled_distillation_weight(
                12, 10, 4, 4, preserve_average=False
            ),
            0.75,
        )

    def test_dense_target_rejects_out_of_range_coords_before_indexing(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            dense_occupancy_from_coords(
                torch.tensor([[0, 4, 0, 0]], dtype=torch.int32),
                batch_size=1,
                resolution=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_slat_consistency_backpropagates_only_through_frozen_inputs(self):
        torch.manual_seed(0)
        trainer = object.__new__(
            ImageConditionedSLatAwareSSFlowMatchingCFGTrainer_ControlNet
        )
        trainer.slat_sigma_min = 1e-5
        trainer.slat_gate_temperature = 1.0
        trainer.slat_gate_floor = 0.0
        trainer.slat_gate_mode = "soft"
        trainer.slat_gate_threshold = 0.5
        trainer.sample_t = lambda _batch_size: torch.tensor([0.5])

        teacher = _FakeTeacher(channels=2).requires_grad_(False).eval()
        decoder = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)
        decoder.weight.data.fill_(1.0)
        decoder.requires_grad_(False).eval()
        trainer.slat_teacher = teacher

        pred_x0 = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
        occupancy_logits = decoder(pred_x0)
        slat_x0 = _FakeSparseTensor(
            coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            feats=torch.tensor([[1.0, -0.5]]),
        )

        loss, _ = trainer._slat_consistency_loss(
            slat_x0,
            occupancy_logits,
            cond=torch.zeros(1, 1),
        )
        loss.backward()

        self.assertIsNotNone(pred_x0.grad)
        self.assertGreater(pred_x0.grad.abs().sum(), 0)
        self.assertIsNone(decoder.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in teacher.parameters())
        )

    def test_straight_through_gate_is_hard_forward_and_soft_backward(self):
        trainer = object.__new__(
            ImageConditionedSLatAwareSSFlowMatchingCFGTrainer_ControlNet
        )
        trainer.slat_gate_temperature = 1.0
        trainer.slat_gate_floor = 0.0
        trainer.slat_gate_mode = "straight_through"
        trainer.slat_gate_threshold = 0.5
        logits = torch.tensor([-1.0, 1.0], requires_grad=True)

        gate = trainer._candidate_gate(logits)
        torch.testing.assert_close(gate, torch.tensor([0.0, 1.0]))
        gate.sum().backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(logits.grad > 0))


if __name__ == "__main__":
    unittest.main()
