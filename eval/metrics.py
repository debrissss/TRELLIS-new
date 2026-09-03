#!/usr/bin/env python3
"""Shared metric helpers for TRELLIS evaluation scripts."""

# 中文说明：
# 图像指标和数值汇总的公共工具模块，不作为独立命令行入口使用。
# 提供 L1、MSE、PSNR、SSIM、LPIPS 以及统计汇总函数。

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_LPIPS_VGG = None


def metric_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def l1_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = mse_metric(pred, target).clamp_min(torch.finfo(pred.dtype).eps)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device, dtype=pred.dtype) / torch.sqrt(mse))


def gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.tensor(values, device=device, dtype=dtype)
    return gauss / gauss.sum()


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channel = img1.size(-3)
    window_1d = gaussian_window(window_size, 1.5, img1.device, img1.dtype).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def lpips_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        from lpips import LPIPS

        _LPIPS_VGG = LPIPS(net="vgg").to(img1.device).eval()
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    return _LPIPS_VGG(img1, img2).mean()


def summarize_numeric_values(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {}
    arr = np.array(finite, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def summarize_metric_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] != ""]
        stats = summarize_numeric_values(values)
        if stats:
            summary[metric] = stats
    return summary
