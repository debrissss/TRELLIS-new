"""
Rebuild a missing shard metadata CSV from existing FaceScape preprocessing outputs.

This is intended for cases where metadata_i.csv was lost but latents_i and
renders_cond_i still exist. Fields that cannot be recovered from outputs alone,
such as the original FaceScape local_path, are left blank by default.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
DEFAULT_FEATURE_MODEL = "dinov2_vitl14_reg"


def discover_sha256s(latents_dir: Path, renders_cond_dir: Path, latent_model: str) -> list[str]:
    sha256s = set()
    latent_model_dir = latents_dir / latent_model
    if latent_model_dir.exists():
        sha256s.update(path.stem for path in latent_model_dir.glob("*.npz"))
    elif latents_dir.exists():
        sha256s.update(path.stem for path in latents_dir.rglob("*.npz"))

    if renders_cond_dir.exists():
        sha256s.update(path.name for path in renders_cond_dir.iterdir() if path.is_dir())

    return sorted(sha256s)


def count_num_voxels(latents_dir: Path, latent_model: str, sha256: str) -> int:
    latent_path = latents_dir / latent_model / f"{sha256}.npz"
    if not latent_path.exists():
        matches = list(latents_dir.rglob(f"{sha256}.npz"))
        if not matches:
            return 0
        latent_path = matches[0]

    try:
        with np.load(latent_path) as data:
            if "coords" in data:
                return int(data["coords"].shape[0])
            if "feats" in data:
                return int(data["feats"].shape[0])
    except Exception as e:
        print(f"[WARN] Failed to read latent {latent_path}: {e}")
    return 0


def load_reference_columns(reference_metadata: Path | None) -> list[str]:
    if reference_metadata is None:
        return [
            "sha256",
            "local_path",
            "captions",
            "rendered",
            "voxelized",
            "num_voxels",
            "cond_rendered",
            f"feature_{DEFAULT_FEATURE_MODEL}",
            f"latent_{DEFAULT_LATENT_MODEL}",
        ]
    reference = pd.read_csv(reference_metadata, nrows=0)
    return list(reference.columns)


def build_metadata(
    shard_dir: Path,
    shard_index: str,
    latent_model: str,
    feature_model: str,
    reference_metadata: Path | None,
) -> pd.DataFrame:
    latents_dir = shard_dir / f"latents_{shard_index}"
    renders_cond_dir = shard_dir / f"renders_cond_{shard_index}"

    if not latents_dir.exists():
        raise FileNotFoundError(f"latents directory not found: {latents_dir}")
    if not renders_cond_dir.exists():
        raise FileNotFoundError(f"renders_cond directory not found: {renders_cond_dir}")

    sha256s = discover_sha256s(latents_dir, renders_cond_dir, latent_model)
    if not sha256s:
        raise FileNotFoundError(f"No sha256 entries found under {latents_dir} or {renders_cond_dir}")

    rows = []
    for sha256 in sha256s:
        latent_path = latents_dir / latent_model / f"{sha256}.npz"
        if not latent_path.exists():
            latent_matches = list(latents_dir.rglob(f"{sha256}.npz"))
            latent_exists = len(latent_matches) > 0
        else:
            latent_exists = True

        cond_exists = (renders_cond_dir / sha256 / "transforms.json").exists()
        rows.append({
            "sha256": sha256,
            "local_path": "",
            "captions": f"Recovered FaceScape sample {sha256}",
            "rendered": cond_exists,
            "voxelized": latent_exists,
            "num_voxels": count_num_voxels(latents_dir, latent_model, sha256),
            "cond_rendered": cond_exists,
            f"feature_{feature_model}": latent_exists,
            f"latent_{latent_model}": latent_exists,
        })

    metadata = pd.DataFrame.from_records(rows)
    columns = load_reference_columns(reference_metadata)
    for column in columns:
        if column not in metadata.columns:
            metadata[column] = ""
    extra_columns = [column for column in metadata.columns if column not in columns]
    return metadata[columns + extra_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild metadata_i.csv from latents_i and renders_cond_i."
    )
    parser.add_argument("--shard_dir", type=Path, default=Path("/root/autodl-tmp/preprocess_facescape_1"),
                        help="Directory containing latents_i and renders_cond_i.")
    parser.add_argument("--shard_index", type=str, default="1",
                        help="Shard suffix index, e.g. 1 for metadata_1.csv.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output metadata path. Defaults to shard_dir/metadata_i.csv.")
    parser.add_argument("--latent_model", type=str, default=DEFAULT_LATENT_MODEL,
                        help="Latent model directory name.")
    parser.add_argument("--feature_model", type=str, default=DEFAULT_FEATURE_MODEL,
                        help="Feature model status column suffix.")
    parser.add_argument("--reference_metadata", type=Path, default=None,
                        help="Optional CSV used only for output column order.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.shard_dir / f"metadata_{args.shard_index}.csv")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    metadata = build_metadata(
        shard_dir=args.shard_dir,
        shard_index=args.shard_index,
        latent_model=args.latent_model,
        feature_model=args.feature_model,
        reference_metadata=args.reference_metadata,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    print(f"[INFO] Wrote {len(metadata)} rows to {output}")
    print(f"[INFO] cond_rendered: {int(metadata['cond_rendered'].sum())}")
    latent_column = f"latent_{args.latent_model}"
    if latent_column in metadata.columns:
        print(f"[INFO] {latent_column}: {int(metadata[latent_column].sum())}")
    print("[WARN] local_path cannot be recovered from latents/renders_cond alone and was left blank.")


if __name__ == "__main__":
    main()
