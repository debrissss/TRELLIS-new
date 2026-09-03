#!/usr/bin/env python3
"""Compare split-stage artifacts with an unsplit Stable3DGen reference run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.io import write_json
from eval.mesh_geometry_metrics import compare_meshes, load_mesh


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--ss_flow_manifest", type=Path, required=True)
    parser.add_argument("--ss_decoder_manifest", type=Path, required=True)
    parser.add_argument("--slat_flow_manifest", type=Path, required=True)
    parser.add_argument("--slat_decoder_manifest", type=Path, required=True)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--reference_repeat_dir",
        type=Path,
        default=None,
        help=(
            "Optional second Stable3DGen run with identical inputs. When supplied, "
            "the report separates split-stage differences from sparse-CUDA "
            "repeatability noise already present in Stable3DGen itself."
        ),
    )
    parser.add_argument(
        "--split_decoder_repeat_manifest",
        type=Path,
        default=None,
        help=(
            "Optional manifest from decoding the exact same split SLat latent "
            "again, used to measure decoder-only repeatability."
        ),
    )
    parser.add_argument("--mesh_point_samples", type=int, default=100_000)
    parser.add_argument("--mesh_metric_seed", type=int, default=0)
    return parser


def _read_row(path: Path, sample_id: str | None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if str(row.get("failed", "")).strip().lower() not in {"true", "1", "yes"}
        ]
    if sample_id is not None:
        rows = [row for row in rows if row.get("sample_id") == sample_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one successful row in {path} "
            f"for sample_id={sample_id!r}, got {len(rows)}"
        )
    return rows[0]


def _load_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} is missing {key!r}; keys={data.files}")
        return np.asarray(data[key])


def _numeric_comparison(
    reference: np.ndarray,
    split: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    shape_equal = reference.shape == split.shape
    if not shape_equal:
        return {
            "passed": False,
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "split_shape": list(split.shape),
        }
    reference64 = reference.astype(np.float64, copy=False)
    split64 = split.astype(np.float64, copy=False)
    delta = np.abs(reference64 - split64)
    exact = bool(np.array_equal(reference, split))
    allclose = bool(np.allclose(reference64, split64, atol=atol, rtol=rtol))
    return {
        "passed": allclose,
        "shape_equal": True,
        "exact": exact,
        "allclose": allclose,
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "p95_abs": float(np.percentile(delta, 95)) if delta.size else 0.0,
        "p99_abs": float(np.percentile(delta, 99)) if delta.size else 0.0,
        "reference_shape": list(reference.shape),
        "split_shape": list(split.shape),
    }


def _exact_comparison(reference: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    shape_equal = reference.shape == split.shape
    exact = shape_equal and bool(np.array_equal(reference, split))
    return {
        "passed": exact,
        "shape_equal": shape_equal,
        "exact": exact,
        "reference_shape": list(reference.shape),
        "split_shape": list(split.shape),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_arrays(directory: Path) -> dict[str, np.ndarray]:
    with Image.open(directory / "cond.png") as image:
        prepared_condition = np.asarray(image).copy()
    return {
        "prepared_condition": prepared_condition,
        "condition_features": _load_npz(
            directory / "condition_features.npz",
            "cond",
        ),
        "negative_condition_features": _load_npz(
            directory / "condition_features.npz",
            "neg_cond",
        ),
        "rng_state_after_ss": _load_npz(
            directory / "rng_state_after_ss.npz",
            "torch_cpu_rng_state",
        ),
        "ss_latent": _load_npz(directory / "ss_latent.npz", "z_s"),
        "ss_coords": _load_npz(directory / "ss_coords.npz", "coords"),
        "slat_coords": _load_npz(directory / "slat_latent.npz", "coords"),
        "slat_normalized_feats": _load_npz(
            directory / "slat_latent.npz",
            "normalized_feats",
        ),
        "slat_decoder_ready_feats": _load_npz(
            directory / "slat_latent.npz",
            "feats",
        ),
    }


def _repeatability_report(
    *,
    reference_dir: Path,
    repeat_dir: Path,
    split_arrays: dict[str, np.ndarray],
    split_mesh_path: Path,
    atol: float,
    rtol: float,
    point_samples: int,
    seed: int,
) -> dict[str, Any]:
    reference = _load_reference_arrays(reference_dir)
    repeat = _load_reference_arrays(repeat_dir)
    repeat_checks = {
        "prepared_condition": _exact_comparison(
            reference["prepared_condition"],
            repeat["prepared_condition"],
        ),
        "condition_features": _numeric_comparison(
            reference["condition_features"],
            repeat["condition_features"],
            atol=atol,
            rtol=rtol,
        ),
        "negative_condition_features": _numeric_comparison(
            reference["negative_condition_features"],
            repeat["negative_condition_features"],
            atol=atol,
            rtol=rtol,
        ),
        "rng_state_after_ss": _exact_comparison(
            reference["rng_state_after_ss"],
            repeat["rng_state_after_ss"],
        ),
        "ss_latent": _numeric_comparison(
            reference["ss_latent"],
            repeat["ss_latent"],
            atol=atol,
            rtol=rtol,
        ),
        "ss_coords": _exact_comparison(
            reference["ss_coords"],
            repeat["ss_coords"],
        ),
        "slat_coords": _exact_comparison(
            reference["slat_coords"],
            repeat["slat_coords"],
        ),
        "slat_normalized_feats": _numeric_comparison(
            reference["slat_normalized_feats"],
            repeat["slat_normalized_feats"],
            atol=atol,
            rtol=rtol,
        ),
        "slat_decoder_ready_feats": _numeric_comparison(
            reference["slat_decoder_ready_feats"],
            repeat["slat_decoder_ready_feats"],
            atol=atol,
            rtol=rtol,
        ),
    }
    split_feature_delta = {
        name: _numeric_comparison(
            reference[name],
            split_arrays[name],
            atol=atol,
            rtol=rtol,
        )
        for name in ("slat_normalized_feats", "slat_decoder_ready_feats")
    }

    reference_mesh = load_mesh(reference_dir / "mesh.ply")
    repeat_mesh = load_mesh(repeat_dir / "mesh.ply")
    split_mesh = load_mesh(split_mesh_path)
    stable_repeat_geometry = compare_meshes(
        repeat_mesh,
        reference_mesh,
        point_samples=point_samples,
        seed=seed,
    )
    split_geometry = compare_meshes(
        split_mesh,
        reference_mesh,
        point_samples=point_samples,
        seed=seed,
    )

    ratio_keys = (
        "chamfer_l1",
        "chamfer_l2",
        "pred_to_gt_mean",
        "gt_to_pred_mean",
        "pred_to_gt_p95",
        "gt_to_pred_p95",
    )
    geometry_ratios = {
        key: (
            float(split_geometry[key] / stable_repeat_geometry[key])
            if stable_repeat_geometry[key] != 0
            else None
        )
        for key in ratio_keys
    }
    feature_ratios: dict[str, dict[str, float | None]] = {}
    for name, split_delta in split_feature_delta.items():
        stable_delta = repeat_checks[name]
        feature_ratios[name] = {}
        for metric in ("max_abs", "mean_abs", "p95_abs", "p99_abs"):
            denominator = float(stable_delta[metric])
            feature_ratios[name][metric] = (
                float(split_delta[metric] / denominator)
                if denominator != 0
                else None
            )

    deterministic_prefix = (
        "prepared_condition",
        "condition_features",
        "negative_condition_features",
        "rng_state_after_ss",
        "ss_latent",
        "ss_coords",
        "slat_coords",
    )
    prefix_exact = all(
        bool(repeat_checks[name].get("exact"))
        for name in deterministic_prefix
    )
    slat_repeatable = bool(
        repeat_checks["slat_normalized_feats"].get("exact")
    )
    return {
        "reference_repeat_dir": str(repeat_dir),
        "reference_repeat_checks": repeat_checks,
        "stable3dgen_deterministic_prefix_exact": prefix_exact,
        "stable3dgen_slat_bitwise_repeatable": slat_repeatable,
        "observed_backend_nondeterminism": prefix_exact and not slat_repeatable,
        "slat_feature_delta": {
            "stable_repeat_vs_reference": {
                name: repeat_checks[name]
                for name in (
                    "slat_normalized_feats",
                    "slat_decoder_ready_feats",
                )
            },
            "split_vs_reference": split_feature_delta,
            "split_to_stable_repeat_ratio": feature_ratios,
        },
        "mesh_geometry": {
            "point_samples": point_samples,
            "seed": seed,
            "stable_repeat_vs_reference": stable_repeat_geometry,
            "split_vs_reference": split_geometry,
            "split_to_stable_repeat_ratio": geometry_ratios,
        },
        "interpretation": (
            "The conditioning/RNG/SS prefix repeats bit-for-bit, while the "
            "unsplit Stable3DGen SLat result does not. Therefore a non-zero "
            "SLat or mesh delta from a separately launched split process cannot "
            "by itself establish a split-pipeline mismatch on this sparse-CUDA "
            "backend; compare its magnitude with the Stable3DGen self-repeat."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.reference_repeat_dir = (
        args.reference_repeat_dir.expanduser().resolve()
        if args.reference_repeat_dir is not None
        else None
    )
    args.output_json = args.output_json.expanduser().resolve()
    args.split_decoder_repeat_manifest = (
        args.split_decoder_repeat_manifest.expanduser().resolve()
        if args.split_decoder_repeat_manifest is not None
        else None
    )
    manifests = {
        "ss_flow": args.ss_flow_manifest.expanduser().resolve(),
        "ss_decoder": args.ss_decoder_manifest.expanduser().resolve(),
        "slat_flow": args.slat_flow_manifest.expanduser().resolve(),
        "slat_decoder": args.slat_decoder_manifest.expanduser().resolve(),
    }
    rows = {
        name: _read_row(path, args.sample_id)
        for name, path in manifests.items()
    }
    sample_ids = {row["sample_id"] for row in rows.values()}
    if len(sample_ids) != 1:
        raise ValueError(f"Split manifests refer to different samples: {sample_ids}")
    sample_id = sample_ids.pop()

    reference_cond = np.asarray(Image.open(args.reference_dir / "cond.png"))
    split_cond_path = Path(rows["ss_flow"]["prepared_condition_path"])
    split_cond = np.asarray(Image.open(split_cond_path))

    reference_ss = _load_npz(args.reference_dir / "ss_latent.npz", "z_s")
    split_ss = _load_npz(Path(rows["ss_flow"]["latent_path"]), "z_s")
    reference_condition = _load_npz(
        args.reference_dir / "condition_features.npz",
        "cond",
    )
    split_condition = _load_npz(
        Path(rows["ss_flow"]["condition_features_path"]),
        "cond",
    )
    reference_negative_condition = _load_npz(
        args.reference_dir / "condition_features.npz",
        "neg_cond",
    )
    split_negative_condition = _load_npz(
        Path(rows["ss_flow"]["condition_features_path"]),
        "neg_cond",
    )
    reference_rng_state = _load_npz(
        args.reference_dir / "rng_state_after_ss.npz",
        "torch_cpu_rng_state",
    )
    split_rng_state = _load_npz(
        Path(rows["ss_flow"]["rng_state_path"]),
        "torch_cpu_rng_state",
    )
    reference_ss_coords = _load_npz(args.reference_dir / "ss_coords.npz", "coords")
    split_ss_coords = _load_npz(Path(rows["ss_decoder"]["coords_path"]), "coords")
    reference_slat_coords = _load_npz(
        args.reference_dir / "slat_latent.npz",
        "coords",
    )
    split_slat_path = Path(rows["slat_flow"]["latent_path"])
    split_slat_coords = _load_npz(split_slat_path, "coords")
    reference_slat_feats = _load_npz(
        args.reference_dir / "slat_latent.npz",
        "feats",
    )
    split_slat_feats = _load_npz(split_slat_path, "feats")
    reference_normalized = _load_npz(
        args.reference_dir / "slat_latent.npz",
        "normalized_feats",
    )
    split_normalized = _load_npz(split_slat_path, "normalized_feats")
    split_arrays = {
        "slat_normalized_feats": split_normalized,
        "slat_decoder_ready_feats": split_slat_feats,
    }

    reference_mesh_path = args.reference_dir / "mesh.ply"
    split_mesh_path = Path(rows["slat_decoder"]["artifact_path"])
    reference_mesh = trimesh.load(reference_mesh_path, force="mesh", process=False)
    split_mesh = trimesh.load(split_mesh_path, force="mesh", process=False)

    checks = {
        "prepared_condition": _exact_comparison(reference_cond, split_cond),
        "condition_features": _numeric_comparison(
            reference_condition,
            split_condition,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "negative_condition_features": _numeric_comparison(
            reference_negative_condition,
            split_negative_condition,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "rng_state_after_ss": _exact_comparison(
            reference_rng_state,
            split_rng_state,
        ),
        "ss_latent": _numeric_comparison(
            reference_ss,
            split_ss,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "ss_coords": _exact_comparison(reference_ss_coords, split_ss_coords),
        "slat_coords": _exact_comparison(reference_slat_coords, split_slat_coords),
        "slat_normalized_feats": _numeric_comparison(
            reference_normalized,
            split_normalized,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "slat_decoder_ready_feats": _numeric_comparison(
            reference_slat_feats,
            split_slat_feats,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "mesh_vertices": _numeric_comparison(
            np.asarray(reference_mesh.vertices),
            np.asarray(split_mesh.vertices),
            atol=args.atol,
            rtol=args.rtol,
        ),
        "mesh_faces": _exact_comparison(
            np.asarray(reference_mesh.faces),
            np.asarray(split_mesh.faces),
        ),
    }
    report = {
        "passed": all(check["passed"] for check in checks.values()),
        "sample_id": sample_id,
        "atol": args.atol,
        "rtol": args.rtol,
        "reference_dir": str(args.reference_dir),
        "split_artifacts": {
            "prepared_condition": str(split_cond_path),
            "ss_latent": rows["ss_flow"]["latent_path"],
            "ss_coords": rows["ss_decoder"]["coords_path"],
            "slat_latent": rows["slat_flow"]["latent_path"],
            "mesh": str(split_mesh_path),
        },
        "mesh_file_sha256": {
            "reference": _sha256(reference_mesh_path),
            "split": _sha256(split_mesh_path),
            "exact_file": _sha256(reference_mesh_path) == _sha256(split_mesh_path),
        },
        "checks": checks,
    }
    deterministic_prefix = (
        "prepared_condition",
        "condition_features",
        "negative_condition_features",
        "rng_state_after_ss",
        "ss_latent",
        "ss_coords",
        "slat_coords",
    )
    report["split_deterministic_prefix_exact"] = all(
        bool(checks[name].get("exact"))
        for name in deterministic_prefix
    )
    if args.reference_repeat_dir is not None:
        report["repeatability"] = _repeatability_report(
            reference_dir=args.reference_dir,
            repeat_dir=args.reference_repeat_dir,
            split_arrays=split_arrays,
            split_mesh_path=split_mesh_path,
            atol=args.atol,
            rtol=args.rtol,
            point_samples=args.mesh_point_samples,
            seed=args.mesh_metric_seed,
        )
    if args.split_decoder_repeat_manifest is not None:
        repeat_decoder_row = _read_row(
            args.split_decoder_repeat_manifest,
            sample_id,
        )
        repeat_latent_path = Path(repeat_decoder_row["latent_path"]).resolve()
        repeat_mesh_path = Path(repeat_decoder_row["artifact_path"]).resolve()
        repeat_mesh = load_mesh(repeat_mesh_path)
        report["decoder_repeatability"] = {
            "manifest": str(args.split_decoder_repeat_manifest),
            "same_latent_path": repeat_latent_path == split_slat_path.resolve(),
            "latent_path": str(repeat_latent_path),
            "repeat_mesh_path": str(repeat_mesh_path),
            "mesh_file_sha256": {
                "first": _sha256(split_mesh_path),
                "repeat": _sha256(repeat_mesh_path),
                "exact_file": _sha256(split_mesh_path) == _sha256(repeat_mesh_path),
            },
            "mesh_vertices": _numeric_comparison(
                np.asarray(split_mesh.vertices),
                np.asarray(repeat_mesh.vertices),
                atol=args.atol,
                rtol=args.rtol,
            ),
            "mesh_faces": _exact_comparison(
                np.asarray(split_mesh.faces),
                np.asarray(repeat_mesh.faces),
            ),
        }
    write_json(args.output_json, report)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
