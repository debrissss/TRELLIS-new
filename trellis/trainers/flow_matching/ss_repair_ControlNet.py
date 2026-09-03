"""SS-only repair-aware Flow Matching trainer for FaceScan ControlNet.

The auxiliary objective decodes the predicted clean SS latent with a frozen
SS decoder and supervises four mutually exclusive occupancy regions derived
from the broken control (mesh1) and clean target (mesh2).  No SLat model or
SLat supervision is involved.
"""

from pathlib import Path
from typing import Dict, Tuple

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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=values.dtype)
    count = mask_float.sum()
    return (values * mask_float).sum() / count.clamp_min(1.0)


def repair_region_losses(
    occupancy_logits: torch.Tensor,
    control_occupancy: torch.Tensor,
    target_occupancy: torch.Tensor,
    *,
    background_kernel_size: int = 3,
) -> Dict[str, torch.Tensor]:
    """Return separately normalized fill/remove/keep/background BCE losses."""
    expected_shape = occupancy_logits.shape
    if occupancy_logits.ndim != 5 or occupancy_logits.shape[1] != 1:
        raise ValueError("occupancy_logits must have shape [B, 1, R, R, R]")
    if control_occupancy.shape != expected_shape:
        raise ValueError("control_occupancy must match occupancy_logits shape")
    if target_occupancy.shape != expected_shape:
        raise ValueError("target_occupancy must match occupancy_logits shape")
    if background_kernel_size <= 0 or background_kernel_size % 2 == 0:
        raise ValueError("background_kernel_size must be a positive odd integer")

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

    positive_bce = F.softplus(-logits)
    negative_bce = F.softplus(logits)
    result = {
        "fill": _masked_mean(positive_bce, fill_mask),
        "remove": _masked_mean(negative_bce, remove_mask),
        "keep": _masked_mean(positive_bce, keep_mask),
        "background": _masked_mean(negative_bce, background_mask),
        "fill_voxels": fill_mask.sum().detach(),
        "remove_voxels": remove_mask.sum().detach(),
        "keep_voxels": keep_mask.sum().detach(),
        "background_voxels": background_mask.sum().detach(),
    }
    return result


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
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("repair loss weights must be non-negative")
        if repair_warmup_steps < 0:
            raise ValueError("repair_warmup_steps must be non-negative")
        if background_kernel_size <= 0 or background_kernel_size % 2 == 0:
            raise ValueError("background_kernel_size must be a positive odd integer")
        if not frozen_ss_decoder_ckpt:
            raise ValueError("frozen_ss_decoder_ckpt is required")

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
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
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
        regions = repair_region_losses(
            occupancy_logits,
            control,
            target_occupancy,
            background_kernel_size=self.background_kernel_size,
        )
        repair = (
            self.fill_weight * regions["fill"]
            + self.remove_weight * regions["remove"]
            + self.keep_weight * regions["keep"]
            + self.background_weight * regions["background"]
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
        }
        return export
