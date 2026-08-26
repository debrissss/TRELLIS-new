# SS ControlNet timestep-schedule evaluation

This experiment keeps all eight SS ControlNet blocks active and changes only
their shared inference-time strength. Training, SS Flow loss, and SLat models
remain unchanged.

## Schedule domains

The effective strength is always:

```text
effective_control_scale = base_control_scale * gate
```

The original Flow-t domain uses the actual normalized timestep after
`rescale_t`:

```python
flow_schedule = {
    "name": "smoothstep",
    "domain": "flow_t",
    "full_strength_t": 0.65,
    "min_strength_t": 0.25,
    "min_scale": 0.1,
}
```

Sampling runs from `t=1` to `t=0`. Control is full above
`full_strength_t`, smoothly decays between the thresholds, and stays at
`min_scale` below `min_strength_t`.

The progress domain is independent of `rescale_t`:

```python
progress_schedule = {
    "name": "smoothstep",
    "domain": "progress",
    "full_until": 0.6,
    "fade_until": 0.85,
    "min_scale": 0.1,
}
```

Here progress `0` is the first Euler step and progress `1` is the final Euler
step. Control is full through 60% of the trajectory, fades through 85%, then
stays weak. Omitting `domain` preserves the original `flow_t` behavior.

Every SS sample stores a trace with:

```text
step_index, raw_progress, raw_t, rescaled_t, domain, gate,
effective_control_scale
```

This makes schedules comparable across `rescale_t` and step-count settings.

## Repair evaluation

The FaceScan evaluator fixes `base_control_scale=1.0`, seed, prepared mesh1
control, normal image, checkpoint, and sampler settings. By default it compares:

- `baseline`: fixed scale 1.0;
- `mild`: Flow-t 1.0 to 0.1;
- `release`: Flow-t 1.0 to 0.0;
- `earlier_release`: begins and completes decay earlier.

Add `--include_progress_variant` to also evaluate the progress-domain example.
For a custom matrix, pass `--variants_json` containing a list of objects with a
unique `name` and `schedule`. The list must contain
`{"name": "baseline", "schedule": null}`.

```bash
python fine_tuning/eval_face_scan_ControlNet_ss_timestep_schedule.py \
  --deploy_dir /path/to/deploy \
  --data_dir /path/to/test_split \
  --output_dir /path/to/results \
  --base_control_scale 1.0 \
  --checkpoint_step 4500 \
  --checkpoint_kind raw \
  --include_progress_variant
```

Use `--max_samples 1 --steps 4` for a minimal real-checkpoint smoke run before
launching the full 25/50-step comparison.

Checkpoint step and raw/EMA kind are read from safetensors or deploy metadata
when available. CLI values are fallbacks and are checked for conflicts. The
checkpoint and model config SHA-256 hashes are always recorded.

Task-specific regions are defined as:

```text
fill_region   = mesh2 & ~mesh1
remove_region = mesh1 & ~mesh2
keep_region   = mesh1 & mesh2
```

The evaluator reports `fill_recall`, `remove_success`, `keep_recall`, and a
weighted `repair_score`, in addition to global mesh1/mesh2 overlap metrics.
Samples with an empty task region store `null` for that regional rate and are
excluded from its aggregate mean.

The output explicitly records `slat_loaded=false`, `slat_executed=false`, and
`training_schedule_changed=false`. Training-time schedule matching and
SLat-aware SS distillation are separate follow-up stages and should only be
implemented after inference-only repair gains are established.
