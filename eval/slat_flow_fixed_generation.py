#!/usr/bin/env python3
"""Generate fixed SLat flow samples for checkpoint comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision import utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import datasets, models, trainers  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_grid(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(tensor.detach().float().cpu().clamp(0, 1), path, normalize=False)


def save_slat_ply(dataset, latent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reps = dataset.decode_latent(latent.cuda(), batch_size=1)
    if len(reps) != 1:
        raise RuntimeError(f"Expected one decoded representation, got {len(reps)}.")
    reps[0].save_ply(path)


def select_indices(dataset, num_samples: int, seed: int, indices: list[int] | None = None) -> list[int]:
    if indices is not None:
        selected = indices
    else:
        if num_samples <= 0:
            raise ValueError("--num_samples must be positive")
        rng = random.Random(seed)
        selected = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for index in selected:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Dataset index {index} out of range for len={len(dataset)}")
    return selected


def parse_indices(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_dataset(config: dict[str, Any], data_dir: Path):
    dataset_spec = config["dataset"]
    return getattr(datasets, dataset_spec["name"])(str(data_dir), **dataset_spec["args"])


def build_trainer(config: dict[str, Any], dataset, ckpt_path: Path, output_dir: Path, device: str):
    model_dict = {
        name: getattr(models, spec["name"])(**spec["args"]).cuda()
        for name, spec in config["models"].items()
    }
    trainer_args = dict(config["trainer"]["args"])
    trainer_args["finetune_ckpt"] = {"denoiser": str(ckpt_path)}
    trainer = getattr(trainers, config["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(output_dir),
        load_dir=str(output_dir),
        step=None,
    )
    if device != "cuda":
        raise ValueError("SLat flow generation currently requires CUDA because project models call .cuda().")
    return trainer


def get_sample_id(dataset, index: int) -> str:
    try:
        return str(dataset.instances[index][1])
    except Exception:
        return f"index{index:06d}"


@torch.no_grad()
def generate_fixed_samples(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SPCONV_ALGO") != "native":
        print("[WARN] SPCONV_ALGO is not native; current RTX 5090 environment may FPE with spconv auto.", flush=True)
    set_seed(args.seed)
    config = load_json(args.config)
    dataset = build_dataset(config, args.data_dir)
    selected = select_indices(dataset, args.num_samples, args.seed, parse_indices(args.indices))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config, dataset, args.ckpt, output_dir, args.device)
    sampler = trainer.get_sampler()

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for order, index in enumerate(selected):
        sample_id = get_sample_id(dataset, index)
        sample_dir = output_dir / "samples" / sample_id
        try:
            set_seed(args.seed + order)
            item = dataset[index]
            batch = dataset.collate_fn([item])
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x_0 = batch["x_0"].cuda()
            cond_img = batch["cond"].cuda()
            noise = x_0.replace(torch.randn_like(x_0.feats))
            cond_args = trainer.get_inference_cond(cond_img)
            result = sampler.sample(
                trainer.models["denoiser"],
                noise=noise,
                **cond_args,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                verbose=args.verbose,
            )
            generated = result.samples
            gt_vis = dataset.visualize_sample(x_0)
            gen_vis = dataset.visualize_sample(generated)
            tensor_to_png(cond_img[0], sample_dir / "cond.png")
            tensor_to_png(gen_vis[0], sample_dir / "generated_grid.png")
            tensor_to_png(gt_vis[0], sample_dir / "gt_grid.png")
            save_slat_ply(dataset, generated, sample_dir / "generated.ply")
            save_slat_ply(dataset, x_0, sample_dir / "gt.ply")
            if args.save_npz:
                np.savez_compressed(
                    sample_dir / "generated_latent.npz",
                    coords=generated.coords.detach().cpu().numpy()[:, 1:],
                    feats=generated.feats.detach().float().cpu().numpy(),
                )
            rows.append({
                "label": args.label,
                "sample_id": sample_id,
                "index": index,
                "generated_path": str(sample_dir / "generated_grid.png"),
                "gt_path": str(sample_dir / "gt_grid.png"),
                "cond_path": str(sample_dir / "cond.png"),
                "generated_ply_path": str(sample_dir / "generated.ply"),
                "gt_ply_path": str(sample_dir / "gt.ply"),
                "failed": False,
                "error": "",
            })
        except Exception as exc:
            row = {
                "label": args.label,
                "sample_id": sample_id,
                "index": index,
                "generated_path": "",
                "gt_path": "",
                "cond_path": "",
                "generated_ply_path": "",
                "gt_ply_path": "",
                "failed": True,
                "error": repr(exc),
            }
            rows.append(row)
            failed.append(row)
            if args.fail_on_error:
                raise

    save_grid(torch.stack([torch.tensor(np.asarray(Image.open(row["generated_path"]).convert("RGB")).transpose(2, 0, 1) / 255.0) for row in rows if not row["failed"]]), output_dir / "generated_grid.png") if any(not row["failed"] for row in rows) else None
    write_manifest(output_dir / "manifest.csv", rows)
    summary = {
        "label": args.label,
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "ckpt": str(args.ckpt),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "save_ply": True,
        "num_records": len(rows),
        "failed_count": len(failed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "sample_id",
        "index",
        "generated_path",
        "gt_path",
        "cond_path",
        "generated_ply_path",
        "gt_ply_path",
        "failed",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--indices", default=None, help="Optional comma-separated dataset indices.")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = generate_fixed_samples(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
