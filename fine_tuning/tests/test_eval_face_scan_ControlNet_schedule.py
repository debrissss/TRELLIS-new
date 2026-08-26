import json

import torch
from safetensors.torch import save_file

from fine_tuning.eval_face_scan_ControlNet_ss_scale_sweep import (
    default_schedule_variants,
    load_schedule_variants,
    repair_metrics,
    resolve_checkpoint_provenance,
)


def test_repair_metrics_measure_fill_remove_and_keep_regions():
    mesh1 = torch.zeros(2, 2, 2, dtype=torch.bool)
    mesh2 = torch.zeros_like(mesh1)
    mesh1[0, 0, 0] = True  # keep
    mesh2[0, 0, 0] = True
    mesh2[0, 0, 1] = True  # fill
    mesh1[0, 1, 0] = True  # remove

    perfect = mesh2.clone()
    metrics = repair_metrics(perfect, mesh1, mesh2)
    assert metrics["fill_region_voxels"] == 1
    assert metrics["remove_region_voxels"] == 1
    assert metrics["keep_region_voxels"] == 1
    assert metrics["fill_recall"] == 1.0
    assert metrics["remove_success"] == 1.0
    assert metrics["keep_recall"] == 1.0
    assert metrics["repair_score"] == 1.0

    failed = mesh1.clone()
    metrics = repair_metrics(failed, mesh1, mesh2, weights=(0.5, 0.25, 0.25))
    assert metrics["fill_recall"] == 0.0
    assert metrics["remove_success"] == 0.0
    assert metrics["keep_recall"] == 1.0
    assert metrics["repair_score"] == 0.25


def test_repair_metrics_use_none_for_absent_regions():
    mesh = torch.ones(2, 2, 2, dtype=torch.bool)
    metrics = repair_metrics(mesh, mesh, mesh)
    assert metrics["fill_region_voxels"] == 0
    assert metrics["remove_region_voxels"] == 0
    assert metrics["fill_recall"] is None
    assert metrics["remove_success"] is None
    assert metrics["keep_recall"] == 1.0
    assert metrics["repair_score"] is None


def test_default_variants_hold_scale_fixed_and_optionally_add_progress():
    variants = default_schedule_variants(include_progress=False)
    assert [variant["name"] for variant in variants] == [
        "baseline",
        "mild",
        "release",
        "earlier_release",
    ]
    assert variants[0]["schedule"] is None
    assert all(
        variant["schedule"]["domain"] == "flow_t"
        for variant in variants[1:]
    )
    with_progress = default_schedule_variants(include_progress=True)
    assert with_progress[-1]["name"] == "progress_mild"
    assert with_progress[-1]["schedule"]["domain"] == "progress"


def test_custom_variants_require_a_fixed_baseline(tmp_path):
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(
        json.dumps([{"name": "only", "schedule": None}]),
        encoding="utf-8",
    )
    try:
        load_schedule_variants(variants_path, include_progress=False)
    except ValueError as exc:
        assert "baseline" in str(exc)
    else:
        raise AssertionError("Expected missing baseline to be rejected")


def test_checkpoint_provenance_prefers_metadata_and_hashes_files(tmp_path):
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    checkpoint = ckpts / "ss_flow_ControlNet.safetensors"
    config = ckpts / "ss_flow_ControlNet.json"
    save_file(
        {"weight": torch.ones(1)},
        str(checkpoint),
        metadata={"checkpoint_step": "4700", "checkpoint_kind": "ema"},
    )
    config.write_text('{"name": "fake", "args": {}}', encoding="utf-8")

    provenance = resolve_checkpoint_provenance(
        tmp_path,
        checkpoint_step_fallback=4700,
        checkpoint_kind_fallback="ema",
    )
    assert provenance["checkpoint_step"] == 4700
    assert provenance["checkpoint_kind"] == "ema"
    assert len(provenance["checkpoint_sha256"]) == 64
    assert len(provenance["config_sha256"]) == 64
    assert provenance["metadata_detected"] is True
