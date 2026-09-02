import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchvision import utils

from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device
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
    """Recover the flow-matching x0 estimate without dividing by ``1 - t``."""
    t_view = t.view(-1, *[1 for _ in range(x_t.ndim - 1)])
    sigma_t = sigma_min + (1.0 - sigma_min) * t_view
    return (1.0 - sigma_min) * x_t - sigma_t * pred_v


def sparse_candidate_logits(
    occupancy_logits: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Gather dense SS logits at packed ``[batch, x, y, z]`` coordinates."""
    if occupancy_logits.ndim != 5 or occupancy_logits.shape[1] != 1:
        raise ValueError("occupancy_logits must have shape [B, 1, R, R, R]")
    if len(set(occupancy_logits.shape[2:])) != 1:
        raise ValueError("occupancy_logits must use a cubic spatial grid")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4]")
    indices = coords.to(device=occupancy_logits.device, dtype=torch.long)
    resolution = occupancy_logits.shape[-1]
    if indices.numel() and (
        indices[:, 0].min() < 0
        or indices[:, 0].max() >= occupancy_logits.shape[0]
        or indices[:, 1:].min() < 0
        or indices[:, 1:].max() >= resolution
    ):
        raise ValueError("SLat candidate coordinates are outside the SS grid")
    return occupancy_logits[
        indices[:, 0],
        0,
        indices[:, 1],
        indices[:, 2],
        indices[:, 3],
    ]


def dense_occupancy_from_coords(
    coords: torch.Tensor,
    batch_size: int,
    resolution: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0 or resolution <= 0:
        raise ValueError("batch_size and resolution must be positive")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4]")
    indices = coords.to(device=device, dtype=torch.long)
    if indices.numel() and (
        indices[:, 0].min() < 0
        or indices[:, 0].max() >= batch_size
        or indices[:, 1:].min() < 0
        or indices[:, 1:].max() >= resolution
    ):
        raise ValueError("SLat coordinates are outside the dense occupancy grid")
    target = torch.zeros(
        batch_size,
        1,
        resolution,
        resolution,
        resolution,
        dtype=dtype,
        device=device,
    )
    if indices.numel():
        target[
            indices[:, 0],
            0,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ] = 1.0
    return target


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.float())
    target = target.float()
    reduce_dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target.sum(
        dim=reduce_dims
    )
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def scheduled_distillation_weight(
    step: int,
    start_step: int,
    warmup_steps: int,
) -> float:
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step - start_step + 1) / warmup_steps)


def sampled_distillation_weight(
    step: int,
    start_step: int,
    warmup_steps: int,
    every_n_steps: int,
    *,
    preserve_average: bool,
) -> float:
    """Return the active-step weight for periodically sampled distillation."""
    schedule = scheduled_distillation_weight(step, start_step, warmup_steps)
    if schedule == 0.0 or step % every_n_steps != 0:
        return 0.0
    return schedule * (every_n_steps if preserve_average else 1.0)


class ImageConditionedSLatAwareSSFlowMatchingCFGTrainer_ControlNet(
    ImageConditionedFlowMatchingCFGTrainer_ControlNet
):
    """Task-aware SS distillation with a frozen SLat flow teacher.

    This trainer deliberately keeps the candidate topology fixed to the paired
    ground-truth SLat coordinates. SS probabilities gate the noisy SLat features,
    and a frozen SLat model measures how much that corruption changes its flow
    prediction. The construction is a continuous/straight-through surrogate; it
    does not claim to differentiate through dynamic ``argwhere`` topology.
    """

    def __init__(
        self,
        models,
        dataset,
        *,
        frozen_model_ckpts: Dict[str, str],
        occupancy_bce_weight: float = 0.1,
        occupancy_dice_weight: float = 0.1,
        slat_distill_weight: float = 0.05,
        slat_sigma_min: float = 1e-5,
        slat_gate_temperature: float = 1.0,
        slat_gate_floor: float = 0.05,
        slat_gate_mode: str = "soft",
        slat_gate_threshold: float = 0.5,
        slat_distill_start_step: int = 0,
        slat_distill_warmup_steps: int = 1000,
        slat_distill_every_n_steps: int = 1,
        slat_distill_preserve_average: bool = True,
        dataset_snapshot_num_samples: int = 0,
        **kwargs,
    ):
        required_models = {"denoiser", "ss_decoder", "slat_teacher"}
        missing_models = required_models.difference(models)
        if missing_models:
            raise KeyError(
                "SLat-aware distillation requires model entries: "
                + ", ".join(sorted(missing_models))
            )
        if slat_gate_mode not in {"soft", "straight_through"}:
            raise ValueError(
                "slat_gate_mode must be 'soft' or 'straight_through'"
            )
        if slat_gate_temperature <= 0:
            raise ValueError("slat_gate_temperature must be positive")
        if not 0.0 <= slat_gate_floor <= 1.0:
            raise ValueError("slat_gate_floor must be in [0, 1]")
        if slat_distill_every_n_steps <= 0:
            raise ValueError("slat_distill_every_n_steps must be positive")
        if not 0.0 <= slat_gate_threshold <= 1.0:
            raise ValueError("slat_gate_threshold must be in [0, 1]")
        if not 0.0 <= slat_sigma_min < 1.0:
            raise ValueError("slat_sigma_min must be in [0, 1)")
        if slat_distill_start_step < 0 or slat_distill_warmup_steps < 0:
            raise ValueError("distillation start and warmup steps must be non-negative")
        if any(
            weight < 0
            for weight in (
                occupancy_bce_weight,
                occupancy_dice_weight,
                slat_distill_weight,
            )
        ):
            raise ValueError("all auxiliary loss weights must be non-negative")
        if dataset_snapshot_num_samples < 0:
            raise ValueError("dataset_snapshot_num_samples must be non-negative")

        # Auxiliary models must stay outside BasicTrainer.models: otherwise DDP
        # tries to wrap parameter-free modules and the optimizer/checkpoints gain
        # frozen teacher state. They are reloaded from explicit immutable paths.
        trainable_models = dict(models)
        self.ss_decoder = trainable_models.pop("ss_decoder")
        self.slat_teacher = trainable_models.pop("slat_teacher")
        self._load_frozen_model(
            self.ss_decoder,
            frozen_model_ckpts.get("ss_decoder"),
            "ss_decoder",
        )
        self._load_frozen_model(
            self.slat_teacher,
            frozen_model_ckpts.get("slat_teacher"),
            "slat_teacher",
        )
        self._freeze_auxiliary_model(self.ss_decoder)
        self._freeze_auxiliary_model(self.slat_teacher)

        self.occupancy_bce_weight = float(occupancy_bce_weight)
        self.occupancy_dice_weight = float(occupancy_dice_weight)
        self.slat_distill_weight = float(slat_distill_weight)
        self.slat_sigma_min = float(slat_sigma_min)
        self.slat_gate_temperature = float(slat_gate_temperature)
        self.slat_gate_floor = float(slat_gate_floor)
        self.slat_gate_mode = slat_gate_mode
        self.slat_gate_threshold = float(slat_gate_threshold)
        self.slat_distill_start_step = int(slat_distill_start_step)
        self.slat_distill_warmup_steps = int(slat_distill_warmup_steps)
        self.slat_distill_every_n_steps = int(slat_distill_every_n_steps)
        self.slat_distill_preserve_average = bool(
            slat_distill_preserve_average
        )
        self.dataset_snapshot_num_samples = int(dataset_snapshot_num_samples)

        super().__init__(trainable_models, dataset, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        # Paired SLat targets are training-only. In particular, do not pass a
        # SparseTensor through the sampler to the SS denoiser during snapshots.
        kwargs.pop("slat_x_0", None)
        return super().get_inference_cond(cond, **kwargs)

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """Visualize a bounded batch without copying paired SLat to the GPU."""
        num_samples = min(num_samples, self.dataset_snapshot_num_samples)
        if num_samples == 0:
            if self.is_master:
                print("Skipping dataset snapshot for SLat-aware training.")
            return

        dataloader = DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=(
                self.dataset.collate_fn
                if hasattr(self.dataset, "collate_fn")
                else None
            ),
        )
        data = next(iter(dataloader))
        if not isinstance(data, dict):
            raise TypeError("dataset snapshot expects an unsplit dict batch")
        data.pop("slat_x_0", None)
        data = recursive_to_device(data, self.device)

        # Reuse the already-loaded frozen decoder instead of temporarily
        # allocating a second decoder inside the dataset visualizer.
        previous_decoder = getattr(self.dataset, "ss_dec", None)
        self.dataset.ss_dec = self.ss_decoder
        try:
            vis = self.visualize_sample(data)
        finally:
            self.dataset.ss_dec = previous_decoder

        save_cfg = (
            [(f"dataset_{key}", value) for key, value in vis.items()]
            if isinstance(vis, dict)
            else [("dataset", vis)]
        )
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=max(1, int(np.sqrt(num_samples))),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @staticmethod
    def _load_frozen_model(model, checkpoint: Optional[str], name: str):
        if not checkpoint:
            raise ValueError(f"Missing frozen checkpoint for {name}")
        checkpoint_path = checkpoint
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            # Loading the 1.2 GB SLat teacher directly avoids broadcasting a
            # second full byte copy through GPU memory on multi-GPU jobs.
            state_dict = load_file(
                checkpoint_path, device=str(model.device)
            )
        else:
            checkpoint_data = read_file_dist(checkpoint_path)
            state_dict = torch.load(
                checkpoint_data,
                map_location=model.device,
                weights_only=True,
            )
        model.load_state_dict(state_dict)

    @staticmethod
    def _freeze_auxiliary_model(model):
        model.requires_grad_(False)
        model.eval()

    def _candidate_gate(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(
            logits.float() / self.slat_gate_temperature
        )
        if self.slat_gate_mode == "straight_through":
            hard = (probabilities >= self.slat_gate_threshold).to(
                probabilities.dtype
            )
            probabilities = probabilities + (hard - probabilities).detach()
        return self.slat_gate_floor + (
            1.0 - self.slat_gate_floor
        ) * probabilities

    def _slat_consistency_loss(
        self,
        slat_x_0: sp.SparseTensor,
        occupancy_logits: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = slat_x_0.shape[0]
        t = self.sample_t(batch_size).to(slat_x_0.device).float()
        per_voxel_t = t[slat_x_0.coords[:, 0].long(), None]
        sigma_t = self.slat_sigma_min + (
            1.0 - self.slat_sigma_min
        ) * per_voxel_t
        noise = torch.randn_like(slat_x_0.feats)
        noisy_feats = (
            (1.0 - per_voxel_t) * slat_x_0.feats + sigma_t * noise
        )
        clean_input = slat_x_0.replace(noisy_feats)

        candidate_logits = sparse_candidate_logits(
            occupancy_logits, slat_x_0.coords
        )
        gate = self._candidate_gate(candidate_logits).to(noisy_feats.dtype)
        gated_input = clean_input.replace(noisy_feats * gate[:, None])

        # The clean branch is the immutable teacher target. The gated branch
        # keeps autograd enabled for its input while all teacher parameters stay
        # frozen, so gradients terminate at the SS-derived gate.
        with torch.no_grad():
            clean_prediction = self.slat_teacher(
                clean_input, t * 1000.0, cond
            ).feats
        gated_prediction = self.slat_teacher(
            gated_input, t * 1000.0, cond
        ).feats
        consistency = F.mse_loss(
            gated_prediction.float(), clean_prediction.float()
        )
        return consistency, gate.detach().mean()

    def training_losses(
        self,
        x_0: torch.Tensor,
        slat_x_0: sp.SparseTensor,
        cond=None,
        control=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
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
        target_v = self.get_v(x_0, noise, t)
        mse = F.mse_loss(pred_v, target_v)

        pred_x_0 = flow_velocity_to_x0(
            x_t, pred_v, t, self.sigma_min
        )
        occupancy_logits = self.ss_decoder(pred_x_0).float()
        resolution = occupancy_logits.shape[-1]
        occupancy_target = dense_occupancy_from_coords(
            slat_x_0.coords,
            x_0.shape[0],
            resolution,
            dtype=occupancy_logits.dtype,
            device=occupancy_logits.device,
        )
        occupancy_bce = F.binary_cross_entropy_with_logits(
            occupancy_logits, occupancy_target
        )
        occupancy_dice = soft_dice_loss(
            occupancy_logits, occupancy_target
        )

        active_slat_weight = sampled_distillation_weight(
            self.step,
            self.slat_distill_start_step,
            self.slat_distill_warmup_steps,
            self.slat_distill_every_n_steps,
            preserve_average=self.slat_distill_preserve_average,
        )
        should_distill = active_slat_weight > 0.0
        if should_distill:
            slat_consistency, gate_mean = self._slat_consistency_loss(
                slat_x_0, occupancy_logits, image_cond
            )
        else:
            slat_consistency = occupancy_logits.new_zeros(())
            gate_mean = occupancy_logits.new_zeros(())

        weighted_slat = (
            self.slat_distill_weight
            * active_slat_weight
            * slat_consistency
        )
        total = (
            mse
            + self.occupancy_bce_weight * occupancy_bce
            + self.occupancy_dice_weight * occupancy_dice
            + weighted_slat
        )
        terms = edict(
            mse=mse,
            occupancy_bce=occupancy_bce,
            occupancy_dice=occupancy_dice,
            slat_consistency=slat_consistency,
            slat_weight=active_slat_weight,
            slat_gate_mean=gate_mean,
            loss=total,
        )

        mse_per_instance = np.array(
            [
                F.mse_loss(pred_v[index], target_v[index]).item()
                for index in range(x_0.shape[0])
            ]
        )
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
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
            "occupancy_bce": self.occupancy_bce_weight
            * export["occupancy_bce"],
            "occupancy_dice": self.occupancy_dice_weight
            * export["occupancy_dice"],
            "slat_consistency": self.slat_distill_weight
            * export["slat_weight"]
            * export["slat_consistency"],
        }
        return export
