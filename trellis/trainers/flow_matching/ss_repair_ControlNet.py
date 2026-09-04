"""SS-only repair-aware Flow Matching trainer for FaceScan ControlNet.

The auxiliary objective decodes the predicted clean SS latent with a frozen
SS decoder and supervises four mutually exclusive occupancy regions derived
from the broken control (mesh1) and clean target (mesh2).  No SLat model or
SLat supervision is involved.
"""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...utils.dist_utils import read_file_dist
from .flow_matching_ControlNet import (
    ImageConditionedFlowMatchingCFGTrainer_ControlNet,
)


def flow_velocity_to_x0(
    x_t: torch.Tensor,
    pred_v: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """Recover the clean-latent estimate without dividing by ``1 - t``."""
    if x_t.shape != pred_v.shape:
        raise ValueError("x_t and pred_v must have identical shapes")
    if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
        raise ValueError("t must have shape [batch]")
    t_view = t.view(-1, *[1 for _ in range(x_t.ndim - 1)])
    sigma_t = sigma_min + (1.0 - sigma_min) * t_view
    return (1.0 - sigma_min) * x_t - sigma_t * pred_v


def repair_warmup_scale(step: int, warmup_steps: int) -> float:
    """Linearly enable the repair objective while keeping Flow MSE unchanged."""
    if step < 0 or warmup_steps < 0:
        raise ValueError("step and warmup_steps must be non-negative")
    if warmup_steps == 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def repair_timestep_gate(
    t: torch.Tensor,
    full_until_t: float,
    fade_until_t: float,
) -> torch.Tensor:
    """Weight repair supervision toward the clean, low-t end of the flow."""
    if t.ndim != 1:
        raise ValueError("t must have shape [batch]")
    if not 0.0 <= full_until_t < fade_until_t <= 1.0:
        raise ValueError(
            "repair timesteps must satisfy "
            "0 <= full_until_t < fade_until_t <= 1"
        )
    progress = ((t.float() - full_until_t) / (
        fade_until_t - full_until_t
    )).clamp(0.0, 1.0)
    smoothstep = progress.square() * (3.0 - 2.0 * progress)
    return 1.0 - smoothstep


def mix_low_t_samples(
    base_t: torch.Tensor,
    probability: float,
    max_t: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace a subset of base timesteps with samples from ``U(0,max_t)``."""
    if base_t.ndim != 1:
        raise ValueError("base_t must have shape [batch]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("low_t_sample_probability must be in [0, 1]")
    if not 0.0 < max_t <= 1.0:
        raise ValueError("low_t_sample_max must be in (0, 1]")
    choose_low_t = torch.rand_like(base_t) < probability
    low_t = torch.rand_like(base_t) * max_t
    return torch.where(choose_low_t, low_t, base_t), choose_low_t


def flow_euler_time_sequence(
    steps: int,
    rescale_t: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the exact rescaled time grid used by ``FlowEulerSampler``."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if rescale_t <= 0:
        raise ValueError("rescale_t must be positive")
    raw_t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    return rescale_t * raw_t / (1.0 + (rescale_t - 1.0) * raw_t)


def flow_euler_terminal_unroll(
    model,
    x_t: torch.Tensor,
    cond: torch.Tensor,
    t_sequence: torch.Tensor,
    *,
    prepared_control: torch.Tensor,
    control_scale: float,
    model_kwargs: Dict | None = None,
) -> torch.Tensor:
    """Differentiably unroll a descending tail of the Flow Euler trajectory."""
    if t_sequence.ndim != 1 or t_sequence.numel() < 2:
        raise ValueError("t_sequence must contain at least two scalar timesteps")
    if not bool(torch.all(t_sequence[:-1] > t_sequence[1:])):
        raise ValueError("t_sequence must be strictly descending")
    if control_scale < 0:
        raise ValueError("control_scale must be non-negative")
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    sample = x_t
    for current_t, next_t in zip(t_sequence[:-1], t_sequence[1:]):
        batch_t = current_t.expand(sample.shape[0])
        pred_v = model(
            sample,
            batch_t * 1000.0,
            cond,
            prepared_control=prepared_control,
            control_scale=control_scale,
            **kwargs,
        )
        if pred_v.shape != sample.shape:
            raise ValueError("denoiser prediction must match the terminal sample")
        sample = sample - (current_t - next_t) * pred_v
    return sample


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average each instance separately, then apply optional timestep weights."""
    if values.shape != mask.shape or values.ndim < 2:
        raise ValueError("values and mask must have matching batched shapes")
    mask_float = mask.to(dtype=values.dtype)
    reduce_dims = tuple(range(1, values.ndim))
    count = mask_float.sum(dim=reduce_dims)
    per_instance = (values * mask_float).sum(dim=reduce_dims) / count.clamp_min(1.0)
    valid = (count > 0).to(dtype=values.dtype)
    if sample_weights is None:
        weights = valid
    else:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != values.shape[0]:
            raise ValueError("sample_weights must have shape [batch]")
        weights = sample_weights.to(device=values.device, dtype=values.dtype) * valid
    return (per_instance * weights).mean()


def repair_region_losses(
    occupancy_logits: torch.Tensor,
    control_occupancy: torch.Tensor,
    target_occupancy: torch.Tensor,
    *,
    background_kernel_size: int = 3,
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Return region-balanced, logit-margin occupancy losses."""
    expected_shape = occupancy_logits.shape
    if occupancy_logits.ndim != 5 or occupancy_logits.shape[1] != 1:
        raise ValueError("occupancy_logits must have shape [B, 1, R, R, R]")
    if control_occupancy.shape != expected_shape:
        raise ValueError("control_occupancy must match occupancy_logits shape")
    if target_occupancy.shape != expected_shape:
        raise ValueError("target_occupancy must match occupancy_logits shape")
    if background_kernel_size <= 0 or background_kernel_size % 2 == 0:
        raise ValueError("background_kernel_size must be a positive odd integer")
    if positive_margin < 0 or negative_margin < 0:
        raise ValueError("occupancy logit margins must be non-negative")

    logits = occupancy_logits.float()
    control = control_occupancy > 0.5
    target = target_occupancy > 0.5
    fill_mask = target & ~control
    keep_mask = target & control
    remove_mask = ~target & control
    dilated_target = F.max_pool3d(
        target.float(),
        kernel_size=background_kernel_size,
        stride=1,
        padding=background_kernel_size // 2,
    ) > 0.5
    # Excluding control voxels makes all four repair regions mutually exclusive.
    background_mask = dilated_target & ~target & ~control

    positive_bce = F.softplus(float(positive_margin) - logits)
    negative_bce = F.softplus(float(negative_margin) + logits)
    result = {
        "fill": _masked_mean(positive_bce, fill_mask, sample_weights),
        "remove": _masked_mean(negative_bce, remove_mask, sample_weights),
        "keep": _masked_mean(positive_bce, keep_mask, sample_weights),
        "background": _masked_mean(
            negative_bce, background_mask, sample_weights
        ),
        "fill_voxels": fill_mask.sum().detach(),
        "remove_voxels": remove_mask.sum().detach(),
        "keep_voxels": keep_mask.sum().detach(),
        "background_voxels": background_mask.sum().detach(),
    }
    return result


def _soft_tversky_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    sample_weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    reduce_dims = tuple(range(1, probabilities.ndim))
    roi_float = roi.to(dtype=probabilities.dtype)
    target_float = target.to(dtype=probabilities.dtype)
    true_positive = (probabilities * target_float * roi_float).sum(reduce_dims)
    false_positive = (
        probabilities * (1.0 - target_float) * roi_float
    ).sum(reduce_dims)
    false_negative = (
        (1.0 - probabilities) * target_float * roi_float
    ).sum(reduce_dims)
    loss = 1.0 - (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    valid = (roi_float.sum(reduce_dims) > 0).to(dtype=probabilities.dtype)
    return (loss * sample_weights.to(probabilities.dtype) * valid).mean()


def repair_v2_losses(
    occupancy_logits: torch.Tensor,
    control_occupancy: torch.Tensor,
    target_occupancy: torch.Tensor,
    *,
    sample_weights: torch.Tensor,
    fill_weight: float,
    remove_weight: float,
    keep_weight: float,
    background_weight: float,
    background_kernel_size: int,
    positive_margin: float,
    negative_margin: float,
    tversky_alpha: float,
    tversky_beta: float,
    repair_roi_kernel_size: int,
    coverage_kernel_size: int,
    multiscale_factors: Sequence[int],
    closing_kernel_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute the low-t, region-aware SS occupancy repair-v2 objectives."""
    odd_kernels = {
        "repair_roi_kernel_size": repair_roi_kernel_size,
        "coverage_kernel_size": coverage_kernel_size,
        "closing_kernel_size": closing_kernel_size,
    }
    for name, size in odd_kernels.items():
        if size <= 0 or size % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    if tversky_alpha < 0 or tversky_beta < 0:
        raise ValueError("Tversky alpha and beta must be non-negative")
    if tversky_alpha + tversky_beta <= 0:
        raise ValueError("At least one Tversky coefficient must be positive")
    factors = tuple(int(factor) for factor in multiscale_factors)
    if any(factor <= 1 for factor in factors):
        raise ValueError("multiscale_factors must contain integers greater than 1")

    control = control_occupancy > 0.5
    target = target_occupancy > 0.5
    fill_mask = target & ~control
    remove_mask = ~target & control
    repair_seed = fill_mask | remove_mask
    roi = F.max_pool3d(
        repair_seed.float(),
        kernel_size=repair_roi_kernel_size,
        stride=1,
        padding=repair_roi_kernel_size // 2,
    ) > 0.5
    probabilities = torch.sigmoid(occupancy_logits.float())

    regions = repair_region_losses(
        occupancy_logits,
        control_occupancy,
        target_occupancy,
        background_kernel_size=background_kernel_size,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
        sample_weights=sample_weights,
    )
    margin = (
        fill_weight * regions["fill"]
        + remove_weight * regions["remove"]
        + keep_weight * regions["keep"]
        + background_weight * regions["background"]
    )
    tversky = _soft_tversky_loss(
        probabilities,
        target,
        roi,
        alpha=tversky_alpha,
        beta=tversky_beta,
        sample_weights=sample_weights,
    )

    local_probability = F.max_pool3d(
        probabilities,
        kernel_size=coverage_kernel_size,
        stride=1,
        padding=coverage_kernel_size // 2,
    )
    coverage = _masked_mean(
        -torch.log(local_probability.clamp_min(1e-6)),
        fill_mask,
        sample_weights,
    )

    multiscale = probabilities.new_zeros(())
    for factor in factors:
        pooled_probability = F.max_pool3d(
            probabilities, kernel_size=factor, stride=factor
        )
        pooled_target = F.max_pool3d(
            target.float(), kernel_size=factor, stride=factor
        ) > 0.5
        pooled_control = F.max_pool3d(
            control.float(), kernel_size=factor, stride=factor
        ) > 0.5
        pooled_seed = pooled_target ^ pooled_control
        pooled_roi = F.max_pool3d(
            pooled_seed.float(), kernel_size=3, stride=1, padding=1
        ) > 0.5
        multiscale = multiscale + (1.0 / factor) * _soft_tversky_loss(
            pooled_probability,
            pooled_target,
            pooled_roi,
            alpha=tversky_alpha,
            beta=tversky_beta,
            sample_weights=sample_weights,
        )

    dilated_probability = F.max_pool3d(
        probabilities,
        kernel_size=closing_kernel_size,
        stride=1,
        padding=closing_kernel_size // 2,
    )
    closed_probability = -F.max_pool3d(
        -dilated_probability,
        kernel_size=closing_kernel_size,
        stride=1,
        padding=closing_kernel_size // 2,
    )
    fill_roi = F.max_pool3d(
        fill_mask.float(),
        kernel_size=repair_roi_kernel_size,
        stride=1,
        padding=repair_roi_kernel_size // 2,
    ) > 0.5
    closing = _masked_mean(
        (closed_probability - probabilities).abs(),
        fill_roi,
        sample_weights,
    )

    return {
        **regions,
        "margin": margin,
        "tversky": tversky,
        "coverage": coverage,
        "multiscale": multiscale,
        "closing": closing,
    }


class ImageConditionedSSRepairFlowMatchingCFGTrainer_ControlNet(
    ImageConditionedFlowMatchingCFGTrainer_ControlNet
):
    """Train only SS ControlNet with a decoded, region-balanced repair loss."""

    def __init__(
        self,
        models,
        dataset,
        *,
        frozen_ss_decoder_ckpt: str,
        repair_weight: float = 0.05,
        repair_warmup_steps: int = 500,
        fill_weight: float = 1.0,
        remove_weight: float = 1.0,
        keep_weight: float = 0.5,
        background_weight: float = 0.5,
        background_kernel_size: int = 3,
        repair_full_until_t: float = 0.2,
        repair_fade_until_t: float = 0.5,
        low_t_sample_probability: float = 0.3,
        low_t_sample_max: float = 0.4,
        positive_margin: float = 1.5,
        negative_margin: float = 1.0,
        tversky_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        repair_roi_kernel_size: int = 5,
        coverage_weight: float = 0.5,
        coverage_kernel_size: int = 3,
        multiscale_weight: float = 0.5,
        multiscale_factors: Sequence[int] = (2, 4),
        closing_weight: float = 0.05,
        closing_kernel_size: int = 3,
        **kwargs,
    ):
        if "denoiser" not in models or "ss_decoder" not in models:
            raise KeyError("repair-aware training requires denoiser and ss_decoder")
        weights = {
            "repair_weight": repair_weight,
            "fill_weight": fill_weight,
            "remove_weight": remove_weight,
            "keep_weight": keep_weight,
            "background_weight": background_weight,
            "tversky_weight": tversky_weight,
            "coverage_weight": coverage_weight,
            "multiscale_weight": multiscale_weight,
            "closing_weight": closing_weight,
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("repair loss weights must be non-negative")
        if repair_warmup_steps < 0:
            raise ValueError("repair_warmup_steps must be non-negative")
        if background_kernel_size <= 0 or background_kernel_size % 2 == 0:
            raise ValueError("background_kernel_size must be a positive odd integer")
        if not frozen_ss_decoder_ckpt:
            raise ValueError("frozen_ss_decoder_ckpt is required")
        # Validate scalar schedule arguments once at startup.
        repair_timestep_gate(
            torch.zeros(1), repair_full_until_t, repair_fade_until_t
        )
        mix_low_t_samples(
            torch.zeros(1), low_t_sample_probability, low_t_sample_max
        )

        trainable_models = dict(models)
        self.ss_decoder = trainable_models.pop("ss_decoder")
        self._load_frozen_ss_decoder(frozen_ss_decoder_ckpt)
        self.ss_decoder.requires_grad_(False)
        self.ss_decoder.eval()

        self.repair_weight = float(repair_weight)
        self.repair_warmup_steps = int(repair_warmup_steps)
        self.fill_weight = float(fill_weight)
        self.remove_weight = float(remove_weight)
        self.keep_weight = float(keep_weight)
        self.background_weight = float(background_weight)
        self.background_kernel_size = int(background_kernel_size)
        self.repair_full_until_t = float(repair_full_until_t)
        self.repair_fade_until_t = float(repair_fade_until_t)
        self.low_t_sample_probability = float(low_t_sample_probability)
        self.low_t_sample_max = float(low_t_sample_max)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.repair_roi_kernel_size = int(repair_roi_kernel_size)
        self.coverage_weight = float(coverage_weight)
        self.coverage_kernel_size = int(coverage_kernel_size)
        self.multiscale_weight = float(multiscale_weight)
        self.multiscale_factors = tuple(int(x) for x in multiscale_factors)
        self.closing_weight = float(closing_weight)
        self.closing_kernel_size = int(closing_kernel_size)

        super().__init__(trainable_models, dataset, **kwargs)

    def _load_frozen_ss_decoder(self, checkpoint: str) -> None:
        checkpoint_path = Path(checkpoint)
        device = next(self.ss_decoder.parameters()).device
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path), device=str(device))
        else:
            state_dict = torch.load(
                read_file_dist(str(checkpoint_path)),
                map_location=device,
                weights_only=True,
            )
        self.ss_decoder.load_state_dict(state_dict)

    def get_inference_cond(self, cond, target_occupancy=None, **kwargs):
        # target_occupancy is training-only and must never reach the SS sampler.
        return super().get_inference_cond(cond, **kwargs)

    def training_losses(
        self,
        x_0: torch.Tensor,
        target_occupancy: torch.Tensor,
        cond=None,
        control=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        if control is None:
            raise ValueError("repair-aware training requires control occupancy")

        noise = torch.randn_like(x_0)
        base_t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        t, low_t_selected = mix_low_t_samples(
            base_t,
            self.low_t_sample_probability,
            self.low_t_sample_max,
        )
        x_t = self.diffuse(x_0, t, noise=noise)
        image_cond = self.get_cond(cond, **kwargs)
        pred_v = self.training_models["denoiser"](
            x_t,
            t * 1000.0,
            image_cond,
            control=control,
            **kwargs,
        )
        if pred_v.shape != noise.shape or pred_v.shape != x_0.shape:
            raise ValueError("denoiser prediction must match x_0 shape")
        target_v = self.get_v(x_0, noise, t)
        mse = F.mse_loss(pred_v, target_v)

        pred_x_0 = flow_velocity_to_x0(x_t, pred_v, t, self.sigma_min)
        # Decoder parameters are frozen, but this forward must remain under
        # autograd so occupancy losses can update the SS ControlNet branch.
        decoder_dtype = next(self.ss_decoder.parameters()).dtype
        occupancy_logits = self.ss_decoder(
            pred_x_0.to(dtype=decoder_dtype)
        ).float()
        repair_gate = repair_timestep_gate(
            t, self.repair_full_until_t, self.repair_fade_until_t
        )
        regions = repair_v2_losses(
            occupancy_logits,
            control,
            target_occupancy,
            sample_weights=repair_gate,
            fill_weight=self.fill_weight,
            remove_weight=self.remove_weight,
            keep_weight=self.keep_weight,
            background_weight=self.background_weight,
            background_kernel_size=self.background_kernel_size,
            positive_margin=self.positive_margin,
            negative_margin=self.negative_margin,
            tversky_alpha=self.tversky_alpha,
            tversky_beta=self.tversky_beta,
            repair_roi_kernel_size=self.repair_roi_kernel_size,
            coverage_kernel_size=self.coverage_kernel_size,
            multiscale_factors=self.multiscale_factors,
            closing_kernel_size=self.closing_kernel_size,
        )
        repair = (
            regions["margin"]
            + self.tversky_weight * regions["tversky"]
            + self.coverage_weight * regions["coverage"]
            + self.multiscale_weight * regions["multiscale"]
            + self.closing_weight * regions["closing"]
        )
        warmup = repair_warmup_scale(self.step, self.repair_warmup_steps)
        active_repair_weight = self.repair_weight * warmup
        total = mse + active_repair_weight * repair

        terms = edict(
            mse=mse,
            repair=repair,
            repair_fill=regions["fill"],
            repair_remove=regions["remove"],
            repair_keep=regions["keep"],
            repair_background=regions["background"],
            repair_margin=regions["margin"],
            repair_tversky=regions["tversky"],
            repair_coverage=regions["coverage"],
            repair_multiscale=regions["multiscale"],
            repair_closing=regions["closing"],
            repair_timestep_gate=repair_gate.mean().detach(),
            low_t_sample_fraction=low_t_selected.float().mean().detach(),
            repair_warmup=occupancy_logits.new_tensor(warmup),
            active_repair_weight=occupancy_logits.new_tensor(
                active_repair_weight
            ),
            fill_voxels=regions["fill_voxels"],
            remove_voxels=regions["remove_voxels"],
            keep_voxels=regions["keep_voxels"],
            background_voxels=regions["background_voxels"],
            loss=total,
        )

        mse_per_instance = np.array(
            [
                F.mse_loss(pred_v[index], target_v[index]).item()
                for index in range(x_0.shape[0])
            ]
        )
        time_bin = np.digitize(
            t.detach().cpu().numpy(), np.linspace(0, 1, 11)
        ) - 1
        for index in range(10):
            if (time_bin == index).sum() != 0:
                terms[f"bin_{index}"] = {
                    "mse": mse_per_instance[time_bin == index].mean()
                }
        return terms, {}

    def export_loss_for_file(self, loss):
        export = super().export_loss_for_file(loss)
        export["contribution"] = {
            "mse": export["mse"],
            "repair": export["active_repair_weight"] * export["repair"],
            "fill": export["active_repair_weight"]
            * self.fill_weight
            * export["repair_fill"],
            "remove": export["active_repair_weight"]
            * self.remove_weight
            * export["repair_remove"],
            "keep": export["active_repair_weight"]
            * self.keep_weight
            * export["repair_keep"],
            "background": export["active_repair_weight"]
            * self.background_weight
            * export["repair_background"],
            "margin": export["active_repair_weight"]
            * export["repair_margin"],
            "tversky": export["active_repair_weight"]
            * self.tversky_weight
            * export["repair_tversky"],
            "coverage": export["active_repair_weight"]
            * self.coverage_weight
            * export["repair_coverage"],
            "multiscale": export["active_repair_weight"]
            * self.multiscale_weight
            * export["repair_multiscale"],
            "closing": export["active_repair_weight"]
            * self.closing_weight
            * export["repair_closing"],
        }
        return export


class ImageConditionedSSTerminalRepairFlowMatchingCFGTrainer_ControlNet(
    ImageConditionedSSRepairFlowMatchingCFGTrainer_ControlNet
):
    """Supervise decoded occupancy after the real low-t Euler trajectory tail.

    The ordinary Flow Matching MSE remains a random single-timestep objective.
    Only the auxiliary SS occupancy objective is moved to a differentiable
    unroll of the final Euler updates used at inference.
    """

    def __init__(
        self,
        *args,
        terminal_sampler_steps: int = 25,
        terminal_unroll_steps: int = 4,
        terminal_rescale_t: float = 3.0,
        terminal_control_scale: float = 0.75,
        terminal_loss_interval: int = 1,
        **kwargs,
    ):
        if terminal_sampler_steps <= 0:
            raise ValueError("terminal_sampler_steps must be positive")
        if not 1 <= terminal_unroll_steps <= terminal_sampler_steps:
            raise ValueError(
                "terminal_unroll_steps must be in [1, terminal_sampler_steps]"
            )
        if terminal_rescale_t <= 0:
            raise ValueError("terminal_rescale_t must be positive")
        if terminal_control_scale < 0:
            raise ValueError("terminal_control_scale must be non-negative")
        if terminal_loss_interval <= 0:
            raise ValueError("terminal_loss_interval must be positive")
        self.terminal_sampler_steps = int(terminal_sampler_steps)
        self.terminal_unroll_steps = int(terminal_unroll_steps)
        self.terminal_rescale_t = float(terminal_rescale_t)
        self.terminal_control_scale = float(terminal_control_scale)
        self.terminal_loss_interval = int(terminal_loss_interval)
        super().__init__(*args, **kwargs)

    def training_losses(
        self,
        x_0: torch.Tensor,
        target_occupancy: torch.Tensor,
        cond=None,
        control=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        if control is None:
            raise ValueError("terminal repair training requires control occupancy")

        denoiser = self.training_models["denoiser"]
        image_cond = self.get_cond(cond, **kwargs)
        prepared_control = denoiser.prepare_control(
            control, batch_size=x_0.shape[0]
        )

        # Keep the original random-t Flow Matching objective unchanged.
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        pred_v = denoiser(
            x_t,
            t * 1000.0,
            image_cond,
            prepared_control=prepared_control,
            **kwargs,
        )
        if pred_v.shape != noise.shape or pred_v.shape != x_0.shape:
            raise ValueError("denoiser prediction must match x_0 shape")
        target_v = self.get_v(x_0, noise, t)
        mse = F.mse_loss(pred_v, target_v)

        # Match the exact tail of the inference sampler.  Starting from a
        # ground-truth-corrupted latent isolates late-stage correction while
        # avoiding an impractical 25-step training unroll.
        full_t_sequence = flow_euler_time_sequence(
            self.terminal_sampler_steps,
            self.terminal_rescale_t,
            device=x_0.device,
            dtype=torch.float32,
        )
        terminal_t_sequence = full_t_sequence[-(self.terminal_unroll_steps + 1):]
        start_t = terminal_t_sequence[0].expand(x_0.shape[0])
        terminal_loss_active = self.step % self.terminal_loss_interval == 0
        if terminal_loss_active:
            terminal_noise = torch.randn_like(x_0)
            terminal_x_t = self.diffuse(x_0, start_t, noise=terminal_noise)
            terminal_x_0 = flow_euler_terminal_unroll(
                denoiser,
                terminal_x_t,
                image_cond,
                terminal_t_sequence,
                prepared_control=prepared_control,
                control_scale=self.terminal_control_scale,
                model_kwargs=kwargs,
            )

            decoder_dtype = next(self.ss_decoder.parameters()).dtype
            occupancy_logits = self.ss_decoder(
                terminal_x_0.to(dtype=decoder_dtype)
            ).float()
            unit_weights = torch.ones(
                x_0.shape[0], device=x_0.device, dtype=torch.float32
            )
            regions = repair_v2_losses(
                occupancy_logits,
                control,
                target_occupancy,
                sample_weights=unit_weights,
                fill_weight=self.fill_weight,
                remove_weight=self.remove_weight,
                keep_weight=self.keep_weight,
                background_weight=self.background_weight,
                background_kernel_size=self.background_kernel_size,
                positive_margin=self.positive_margin,
                negative_margin=self.negative_margin,
                tversky_alpha=self.tversky_alpha,
                tversky_beta=self.tversky_beta,
                repair_roi_kernel_size=self.repair_roi_kernel_size,
                coverage_kernel_size=self.coverage_kernel_size,
                multiscale_factors=self.multiscale_factors,
                closing_kernel_size=self.closing_kernel_size,
            )
            repair = (
                regions["margin"]
                + self.tversky_weight * regions["tversky"]
                + self.coverage_weight * regions["coverage"]
                + self.multiscale_weight * regions["multiscale"]
                + self.closing_weight * regions["closing"]
            )
        else:
            zero = mse.detach().new_zeros(())
            occupancy_logits = x_0
            regions = {
                key: zero
                for key in (
                    "fill", "remove", "keep", "background", "margin",
                    "tversky", "coverage", "multiscale", "closing",
                    "fill_voxels", "remove_voxels", "keep_voxels",
                    "background_voxels",
                )
            }
            repair = zero
        warmup = repair_warmup_scale(self.step, self.repair_warmup_steps)
        active_repair_weight = (
            self.repair_weight * warmup * self.terminal_loss_interval
            if terminal_loss_active
            else 0.0
        )
        total = mse + active_repair_weight * repair

        terms = edict(
            mse=mse,
            repair=repair,
            repair_fill=regions["fill"],
            repair_remove=regions["remove"],
            repair_keep=regions["keep"],
            repair_background=regions["background"],
            repair_margin=regions["margin"],
            repair_tversky=regions["tversky"],
            repair_coverage=regions["coverage"],
            repair_multiscale=regions["multiscale"],
            repair_closing=regions["closing"],
            terminal_start_t=start_t.mean().detach(),
            terminal_unroll_steps=occupancy_logits.new_tensor(
                float(self.terminal_unroll_steps)
            ),
            terminal_control_scale=occupancy_logits.new_tensor(
                self.terminal_control_scale
            ),
            terminal_loss_active=occupancy_logits.new_tensor(
                float(terminal_loss_active)
            ),
            repair_warmup=occupancy_logits.new_tensor(warmup),
            active_repair_weight=occupancy_logits.new_tensor(
                active_repair_weight
            ),
            fill_voxels=regions["fill_voxels"],
            remove_voxels=regions["remove_voxels"],
            keep_voxels=regions["keep_voxels"],
            background_voxels=regions["background_voxels"],
            loss=total,
        )

        mse_per_instance = np.array(
            [
                F.mse_loss(pred_v[index], target_v[index]).item()
                for index in range(x_0.shape[0])
            ]
        )
        time_bin = np.digitize(
            t.detach().cpu().numpy(), np.linspace(0, 1, 11)
        ) - 1
        for index in range(10):
            if (time_bin == index).sum() != 0:
                terms[f"bin_{index}"] = {
                    "mse": mse_per_instance[time_bin == index].mean()
                }
        return terms, {}
