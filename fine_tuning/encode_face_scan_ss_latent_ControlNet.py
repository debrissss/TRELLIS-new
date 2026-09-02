"""Encode FaceScan target occupancies into SS Flow supervision latents.

Unlike the generic TRELLIS encoder utility, this ControlNet-specific entry
reads only ``target_voxels``.  ``control_voxels`` are never eligible as x_0.
Run this script in the TRELLIS GPU environment after building the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import utils3d
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trellis import models


DEFAULT_LATENT_MODEL = (
    "ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8_step0002000"
)


def _load_target_occupancy(path: Path, resolution: int) -> torch.Tensor:
    position = utils3d.io.read_ply(str(path))[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    if torch.any(coords < 0) or torch.any(coords >= resolution):
        raise ValueError(f"Target voxel coordinates out of bounds: {path}")
    occupancy = torch.zeros(
        1,
        resolution,
        resolution,
        resolution,
        dtype=torch.float32,
    )
    occupancy[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return occupancy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode FaceScan target voxels for SS Flow ControlNet"
    )
    parser.add_argument(
        "--data_dir",
        default="datasets/FaceScan_ControlNet/train",
        help="Prepared FaceScan split root.",
    )
    parser.add_argument(
        "--encoder_run",
        default="outputs/train/ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8",
        help="SS encoder/decoder training output containing config.json.",
    )
    parser.add_argument(
        "--encoder_ckpt",
        default="step0002000",
        help="Encoder checkpoint suffix, for example step0002000.",
    )
    parser.add_argument(
        "--latent_model",
        default=DEFAULT_LATENT_MODEL,
        help="Output latent directory and metadata suffix.",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode target latents that already exist.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    encoder_run = Path(args.encoder_run).resolve()
    metadata_path = data_dir / "metadata.csv"
    target_voxel_dir = data_dir / "target_voxels"
    latent_dir = data_dir / "ss_latents" / args.latent_model
    encoder_path = encoder_run / "ckpts" / f"encoder_{args.encoder_ckpt}.pt"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"SS Encoder checkpoint not found: {encoder_path}")

    with (encoder_run / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    encoder_config = config["models"]["encoder"]
    encoder = getattr(models, encoder_config["name"])(
        **encoder_config["args"]
    ).cuda()
    state_dict = torch.load(
        encoder_path,
        map_location="cpu",
        weights_only=True,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()

    metadata = pd.read_csv(metadata_path)
    latent_column = f"ss_latent_{args.latent_model}"
    if latent_column not in metadata.columns:
        metadata[latent_column] = False
    latent_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    with torch.inference_mode():
        for index, row in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc="Encoding FaceScan target latents",
        ):
            instance = str(row["sha256"])
            target_path = target_voxel_dir / f"{instance}.ply"
            output_path = latent_dir / f"{instance}.npz"
            if output_path.is_file() and not args.overwrite:
                metadata.at[index, latent_column] = True
                continue
            if not target_path.is_file():
                metadata.at[index, latent_column] = False
                failures.append((instance, "target voxel missing"))
                continue
            try:
                occupancy = _load_target_occupancy(
                    target_path,
                    args.resolution,
                )[None].cuda()
                latent = encoder(occupancy, sample_posterior=False)
                if not torch.isfinite(latent).all():
                    raise ValueError("encoder produced non-finite latent")
                np.savez_compressed(
                    output_path,
                    mean=latent[0].float().cpu().numpy(),
                )
                metadata.at[index, latent_column] = True
            except Exception as error:
                metadata.at[index, latent_column] = False
                failures.append((instance, str(error)))

    # Write only after the encoding loop so an interruption cannot mark an
    # unfinished sample as ready for training.
    metadata.to_csv(metadata_path, index=False)
    ready = int(
        metadata[latent_column]
        .map(lambda value: str(value).strip().lower() == "true")
        .sum()
    )
    print(f"Target latents ready: {ready}/{len(metadata)}")
    for instance, reason in failures:
        print(f"FAILED {instance}: {reason}")
    if failures:
        raise RuntimeError(f"Failed to encode {len(failures)} FaceScan samples")


if __name__ == "__main__":
    main()
