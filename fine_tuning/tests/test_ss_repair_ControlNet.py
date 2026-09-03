import torch
import torch.nn.functional as F

from trellis.trainers.flow_matching.ss_repair_ControlNet import (
    ImageConditionedSSRepairFlowMatchingCFGTrainer_ControlNet,
    flow_velocity_to_x0,
    repair_region_losses,
    repair_warmup_scale,
)


def test_flow_velocity_to_x0_recovers_clean_latent():
    sigma_min = 1e-5
    clean = torch.randn(2, 3, 2, 2, 2)
    noise = torch.randn_like(clean)
    t = torch.tensor([0.2, 0.8])
    t_view = t.view(2, 1, 1, 1, 1)
    sigma_t = sigma_min + (1.0 - sigma_min) * t_view
    x_t = (1.0 - t_view) * clean + sigma_t * noise
    velocity = (1.0 - sigma_min) * noise - clean

    recovered = flow_velocity_to_x0(x_t, velocity, t, sigma_min)

    torch.testing.assert_close(recovered, clean)


def test_repair_regions_are_exclusive_and_separately_normalized():
    control = torch.zeros(1, 1, 3, 3, 3)
    target = torch.zeros_like(control)
    control[0, 0, 1, 1, 1] = 1  # keep
    target[0, 0, 1, 1, 1] = 1
    target[0, 0, 1, 1, 2] = 1  # fill
    control[0, 0, 0, 0, 0] = 1  # remove
    logits = torch.zeros_like(control, requires_grad=True)

    losses = repair_region_losses(logits, control, target)

    expected = F.softplus(torch.tensor(0.0))
    for key in ("fill", "remove", "keep", "background"):
        torch.testing.assert_close(losses[key], expected)
    assert losses["fill_voxels"].item() == 1
    assert losses["remove_voxels"].item() == 1
    assert losses["keep_voxels"].item() == 1
    # The 3x3x3 target dilation covers this whole toy grid, then excludes the
    # two target voxels and the separate remove/control site.
    assert losses["background_voxels"].item() == 24

    sum(losses[key] for key in ("fill", "remove", "keep", "background")).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_empty_repair_regions_are_zero_not_nan():
    target = torch.ones(1, 1, 2, 2, 2)
    logits = torch.zeros_like(target)
    losses = repair_region_losses(logits, target, target)

    assert losses["fill"].item() == 0.0
    assert losses["remove"].item() == 0.0
    assert losses["background"].item() == 0.0
    assert torch.isfinite(losses["keep"])


def test_repair_warmup_scale_is_linear_and_bounded():
    assert repair_warmup_scale(0, 500) == 0.0
    assert repair_warmup_scale(250, 500) == 0.5
    assert repair_warmup_scale(500, 500) == 1.0
    assert repair_warmup_scale(1000, 500) == 1.0
    assert repair_warmup_scale(0, 0) == 1.0


def test_invalid_background_kernel_is_rejected():
    grid = torch.zeros(1, 1, 2, 2, 2)
    try:
        repair_region_losses(
            grid,
            grid,
            grid,
            background_kernel_size=2,
        )
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("Expected an even background kernel to fail")


def test_repair_loss_backpropagates_only_through_trainable_denoiser():
    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, x_t, t, cond, control=None, **kwargs):
            assert control is not None
            return x_t * self.scale

    decoder = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)
    decoder.requires_grad_(False)
    denoiser = TinyDenoiser()
    trainer = object.__new__(
        ImageConditionedSSRepairFlowMatchingCFGTrainer_ControlNet
    )
    trainer.training_models = {"denoiser": denoiser}
    trainer.ss_decoder = decoder
    trainer.sigma_min = 1e-5
    trainer.step = 500
    trainer.repair_weight = 0.05
    trainer.repair_warmup_steps = 500
    trainer.fill_weight = 1.0
    trainer.remove_weight = 1.0
    trainer.keep_weight = 0.5
    trainer.background_weight = 0.5
    trainer.background_kernel_size = 3
    trainer.sample_t = lambda batch_size: torch.full((batch_size,), 0.5)
    trainer.get_cond = lambda cond, **kwargs: cond

    x_0 = torch.randn(1, 1, 2, 2, 2)
    control = torch.zeros(1, 1, 2, 2, 2)
    target = torch.zeros_like(control)
    target[0, 0, 0, 0, 0] = 1.0
    terms, _ = trainer.training_losses(
        x_0,
        target,
        cond=torch.zeros(1, 1),
        control=control,
    )
    terms["loss"].backward()

    assert denoiser.scale.grad is not None
    assert torch.isfinite(denoiser.scale.grad)
    assert decoder.weight.grad is None
    torch.testing.assert_close(
        terms["loss"],
        terms["mse"] + 0.05 * terms["repair"],
    )
