#!/usr/bin/env python3
"""Evaluate SLat encoder/decoder checkpoints on a fixed FaceScape subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import utils3d.torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import models  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.renderers import GaussianRenderer  # noqa: E402


_LPIPS_VGG = None


def parse_view_indices(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--view_indices must contain at least one integer")
    return values


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metadata_rows(data_dir: Path) -> list[dict[str, str]]:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No eval rows found in {metadata_path}")
    if "sha256" not in rows[0]:
        raise KeyError(f"metadata.csv must contain sha256 column: {metadata_path}")
    return rows


def load_fixed_view(data_dir: Path, sha: str, view_index: int, image_size: int) -> dict[str, torch.Tensor]:
    transforms_path = data_dir / "renders" / sha / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)
    metadata = load_json(transforms_path)
    frames = metadata.get("frames", [])
    if view_index < 0 or view_index >= len(frames):
        raise IndexError(f"{sha}: view_index {view_index} out of range for {len(frames)} frames")
    frame = frames[view_index]

    fov = frame["camera_angle_x"]
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    extrinsics = torch.inverse(c2w)

    image_path = data_dir / "renders" / sha / frame["file_path"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path)
    alpha = image.getchannel("A")
    image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = alpha.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    alpha_tensor = torch.tensor(np.array(alpha)).float() / 255.0
    return {
        "image": image_tensor,
        "alpha": alpha_tensor,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def load_sparse_feature(
    data_dir: Path,
    sha: str,
    feature_model: str,
    resolution: int,
) -> SparseTensor:
    data_resolution = 64
    feature_path = data_dir / "features" / feature_model / f"{sha}.npz"
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    with np.load(feature_path, allow_pickle=False) as data:
        for key in ("indices", "patchtokens"):
            if key not in data.files:
                raise KeyError(f"{feature_path} missing key {key!r}; keys={data.files}")
        coords = torch.tensor(data["indices"]).int()
        feats = torch.tensor(data["patchtokens"]).float()

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{feature_path}: expected indices shape (N, 3), got {tuple(coords.shape)}")
    if feats.ndim != 2 or coords.shape[0] != feats.shape[0]:
        raise ValueError(
            f"{feature_path}: indices/patchtokens shape mismatch: {tuple(coords.shape)} vs {tuple(feats.shape)}"
        )

    if resolution != data_resolution:
        factor = data_resolution // resolution
        if factor <= 0 or data_resolution % resolution != 0:
            raise ValueError(f"Unsupported eval resolution {resolution}; source feature resolution is {data_resolution}")
        coords = coords // factor
        coords, idx = coords.unique(return_inverse=True, dim=0)
        feats = torch.scatter_reduce(
            torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
            dim=0,
            index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
            src=feats,
            reduce="mean",
        )

    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    return SparseTensor(coords=torch.cat([batch, coords], dim=-1), feats=feats)


def build_models(config: dict[str, Any], device: torch.device, fp16_mode: str | None) -> dict[str, torch.nn.Module]:
    model_dict = {}
    for name, spec in config["models"].items():
        model = getattr(models, spec["name"])(**spec["args"]).to(device)
        model.eval()
        model_dict[name] = model
    return model_dict


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device, fp16_mode: str | None) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    if fp16_mode == "inflat_all" and hasattr(model, "convert_to_fp16"):
        model.convert_to_fp16()


def build_renderer(decoder: torch.nn.Module) -> GaussianRenderer:
    renderer = GaussianRenderer({"near": 0.8, "far": 1.6, "bg_color": (0, 0, 0)})
    renderer.pipe.kernel_size = decoder.rep_config["2d_filter_kernel_size"]
    return renderer


def render_batch(renderer: GaussianRenderer, reps, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
    ret = None
    for i, representation in enumerate(reps):
        render_pack = renderer.render(representation, extrinsics[i], intrinsics[i])
        if ret is None:
            ret = {k: [] for k in list(render_pack.keys()) + ["bg_color"]}
        for key, value in render_pack.items():
            ret[key].append(value)
        ret["bg_color"].append(renderer.bg_color)
    if ret is None:
        raise RuntimeError("Renderer returned no outputs.")
    for key, value in ret.items():
        ret[key] = torch.stack(value, dim=0)
    return ret


def tensor_to_image_array(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(tensor.shape)}")
    return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_eval_images(sample_dir: Path, gt_image: torch.Tensor, rec_image: torch.Tensor) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    diff = torch.abs(rec_image.detach().float() - gt_image.detach().float()).clamp(0, 1)
    Image.fromarray(tensor_to_image_array(gt_image)).save(sample_dir / "gt.png")
    Image.fromarray(tensor_to_image_array(rec_image)).save(sample_dir / "rec.png")
    Image.fromarray(tensor_to_image_array(diff)).save(sample_dir / "diff.png")


def metric_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def l1_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = mse_metric(pred, target)
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


def compute_metrics(
    rec_image: torch.Tensor,
    gt_image: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    lambda_ssim: float,
    lambda_lpips: float,
    lambda_kl: float,
    skip_lpips: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["l1"] = metric_value(l1_metric(rec_image, gt_image))
    metrics["mse"] = metric_value(mse_metric(rec_image, gt_image))
    metrics["psnr"] = metric_value(psnr_metric(rec_image, gt_image))
    metrics["ssim_loss"] = metric_value(1 - ssim_metric(rec_image, gt_image))
    if skip_lpips:
        metrics["lpips"] = math.nan
    else:
        metrics["lpips"] = metric_value(lpips_metric(rec_image, gt_image))
    metrics["kl"] = metric_value(0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1))
    metrics["rec"] = metrics["l1"] + lambda_ssim * metrics["ssim_loss"]
    if not math.isnan(metrics["lpips"]):
        metrics["rec"] += lambda_lpips * metrics["lpips"]
    metrics["loss"] = metrics["rec"] + lambda_kl * metrics["kl"]
    return metrics


def summarize_metrics(rows: list[dict[str, Any]], failed: list[dict[str, str]]) -> dict[str, Any]:
    metric_names = ["loss", "rec", "l1", "mse", "psnr", "ssim_loss", "lpips", "kl"]
    summary: dict[str, Any] = {
        "num_records": len(rows),
        "failed_count": len(failed),
        "metrics": {},
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row and not math.isnan(float(row[metric]))]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std(ddof=0)),
        }
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sha256",
        "view_index",
        "loss",
        "rec",
        "l1",
        "mse",
        "psnr",
        "ssim_loss",
        "lpips",
        "kl",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    dataset_args = config["dataset"]["args"]
    trainer_args = config["trainer"]["args"]
    image_size = int(dataset_args["image_size"])
    feature_model = args.feature_model or dataset_args.get("model", "dinov2_vitl14_reg")
    resolution = int(dataset_args.get("resolution", 64))
    view_indices = parse_view_indices(args.view_indices)
    device = torch.device(args.device)
    if device.type == "cpu" and not args.skip_lpips:
        raise ValueError("LPIPS uses CUDA in this project; pass --skip_lpips for CPU evaluation.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata_rows(args.data_dir)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    model_dict = build_models(config, device, trainer_args.get("fp16_mode"))
    load_checkpoint(model_dict["encoder"], args.encoder_ckpt, device, trainer_args.get("fp16_mode"))
    load_checkpoint(model_dict["decoder"], args.decoder_ckpt, device, trainer_args.get("fp16_mode"))
    renderer = build_renderer(model_dict["decoder"])
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.resolution = image_size

    metric_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for row in rows:
        sha = row["sha256"]
        for view_index in view_indices:
            try:
                view = load_fixed_view(args.data_dir, sha, view_index, image_size)
                feats = load_sparse_feature(args.data_dir, sha, feature_model, resolution).to(device)
                image = view["image"].unsqueeze(0).to(device)
                alpha = view["alpha"].unsqueeze(0).to(device)
                extrinsics = view["extrinsics"].unsqueeze(0).to(device)
                intrinsics = view["intrinsics"].unsqueeze(0).to(device)

                z, mean, logvar = model_dict["encoder"](feats, sample_posterior=args.sample_posterior, return_raw=True)
                reps = model_dict["decoder"](z)
                render_results = render_batch(renderer, reps, extrinsics, intrinsics)
                rec_image = render_results["color"]
                bg_color = render_results["bg_color"][..., None, None]
                gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * bg_color
                metrics = compute_metrics(
                    rec_image,
                    gt_image,
                    mean,
                    logvar,
                    lambda_ssim=float(trainer_args.get("lambda_ssim", 0.0)),
                    lambda_lpips=float(trainer_args.get("lambda_lpips", 0.0)),
                    lambda_kl=float(trainer_args.get("lambda_kl", 0.0)),
                    skip_lpips=args.skip_lpips,
                )
                metric_rows.append({"sha256": sha, "view_index": view_index, **metrics})
                if len(metric_rows) <= args.save_images:
                    save_eval_images(output_dir / "samples" / f"{sha}_view{view_index}", gt_image[0], rec_image[0])
            except Exception as exc:
                failed.append({"sha256": sha, "view_index": str(view_index), "error": repr(exc)})

    summary = summarize_metrics(metric_rows, failed)
    summary.update({
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "encoder_ckpt": str(args.encoder_ckpt),
        "decoder_ckpt": str(args.decoder_ckpt),
        "feature_model": feature_model,
        "view_indices": view_indices,
        "sample_posterior": bool(args.sample_posterior),
        "skip_lpips": bool(args.skip_lpips),
    })

    write_rows_csv(output_dir / "metrics.csv", metric_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metric_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "failed_samples.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed and args.fail_on_error:
        raise RuntimeError(f"{len(failed)} eval records failed; see {output_dir / 'failed_samples.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--encoder_ckpt", type=Path, required=True)
    parser.add_argument("--decoder_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--view_indices", default="0", help="Comma-separated fixed view indices, e.g. 0 or 0,4,8,12.")
    parser.add_argument("--feature_model", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_images", type=int, default=16, help="Save gt/rec/diff images for the first N records.")
    parser.add_argument("--sample_posterior", action="store_true", help="Use stochastic posterior sampling; default uses deterministic z.")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS, useful for CPU smoke tests.")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
