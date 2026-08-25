from typing import *
import math
import numbers
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


def _control_schedule_gate(t: float, schedule: Mapping[str, Any]) -> float:
    """Return the relative ControlNet strength for a normalized Flow timestep."""
    if not isinstance(schedule, Mapping):
        raise TypeError("control_schedule must be a mapping")

    allowed_keys = {
        "name",
        "full_strength_t",
        "min_strength_t",
        "min_scale",
    }
    unknown_keys = set(schedule) - allowed_keys
    if unknown_keys:
        raise ValueError(
            "Unknown control_schedule keys: "
            + ", ".join(sorted(unknown_keys))
        )
    if schedule.get("name") != "smoothstep":
        raise ValueError(
            "control_schedule.name must be 'smoothstep'"
        )

    values = {}
    for key, default in (
        ("full_strength_t", 0.65),
        ("min_strength_t", 0.25),
        ("min_scale", 0.1),
    ):
        value = schedule.get(key, default)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"control_schedule.{key} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"control_schedule.{key} must be finite")
        values[key] = value

    full_strength_t = values["full_strength_t"]
    min_strength_t = values["min_strength_t"]
    min_scale = values["min_scale"]
    if not 0.0 <= min_strength_t < full_strength_t <= 1.0:
        raise ValueError(
            "control_schedule timesteps must satisfy "
            "0 <= min_strength_t < full_strength_t <= 1"
        )
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError("control_schedule.min_scale must be in [0, 1]")

    t = float(t)
    if not math.isfinite(t):
        raise ValueError("Flow timestep must be finite")
    if t >= full_strength_t:
        return 1.0
    if t <= min_strength_t:
        return min_scale

    u = (t - min_strength_t) / (full_strength_t - min_strength_t)
    smoothstep = u * u * (3.0 - 2.0 * u)
    return min_scale + (1.0 - min_scale) * smoothstep


def _scheduled_control_scale(
    control_scale: Any,
    t: float,
    schedule: Mapping[str, Any],
) -> Any:
    """Multiply a scalar or per-layer base scale by the timestep gate."""
    gate = _control_schedule_gate(t, schedule)
    if isinstance(control_scale, torch.Tensor):
        return control_scale * gate
    if isinstance(control_scale, np.ndarray):
        return control_scale * gate
    if isinstance(control_scale, numbers.Real) and not isinstance(
        control_scale, bool
    ):
        return float(control_scale) * gate
    if isinstance(control_scale, (str, bytes)):
        raise TypeError("control_scale must be numeric, not a string")
    try:
        raw_scales = list(control_scale)
    except TypeError as exc:
        raise TypeError(
            "control_scale must be a real scalar or a numeric sequence"
        ) from exc
    scaled = []
    for index, value in enumerate(raw_scales):
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise TypeError(
                    f"control_scale[{index}] must be a scalar tensor"
                )
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"control_scale[{index}] must be a real number"
            )
        scaled.append(value * gate)
    return scaled


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(
        self,
        model,
        x_t,
        t,
        cond=None,
        control_schedule: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        # t 是 rescale_t 变换后的真实 Flow timestep。调度在模型调用前
        # 只改变本 step 的有效 scale，不改变基础 control_scale，也不会
        # 影响未启用 schedule 的原始推理路径。
        if control_schedule is not None:
            if "control_scale" not in kwargs:
                raise ValueError(
                    "control_schedule requires control_scale"
                )
            kwargs["control_scale"] = _scheduled_control_scale(
                kwargs["control_scale"], t, control_schedule
            )
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
