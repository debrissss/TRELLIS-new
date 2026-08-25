# SS ControlNet timestep schedule

SS Flow inference can keep all configured ControlNet blocks active while
reducing their shared influence near the clean end of the Flow trajectory.
The effective scale at a timestep is:

```text
effective_control_scale(t) = base_control_scale * gate(t)
```

The optional `control_schedule` mapping currently supports a smoothstep gate:

```python
control_schedule = {
    "name": "smoothstep",
    "full_strength_t": 0.65,
    "min_strength_t": 0.25,
    "min_scale": 0.1,
}

coords = pipeline.sample_sparse_structure(
    cond,
    prepared_control=prepared_control,
    control_scale=1.0,
    control_schedule=control_schedule,
)
```

The schedule uses the normalized Flow timestep after `rescale_t` has been
applied. Sampling runs from `t=1` to `t=0`:

- `t >= full_strength_t`: gate is `1.0`.
- Between both thresholds: gate decays with smoothstep interpolation.
- `t <= min_strength_t`: gate is `min_scale`.

Omitting `control_schedule` preserves fixed-scale inference. A scalar base
scale applies to every configured ControlNet block. A per-block scale sequence
is also supported; the timestep gate multiplies every entry without changing
which blocks are injected.

## FaceScan comparison

Use the existing SS-only evaluator with identical data, checkpoint, seed, and
sampler settings. Keep `--control_scales 1.0` for all three runs.

```text
# Baseline: omit --control_schedule_min_scale
--control_scales 1.0

# Mild late-stage control
--control_scales 1.0 --control_schedule_min_scale 0.1

# Strong late-stage release
--control_scales 1.0 --control_schedule_min_scale 0.0
```

The evaluator records the resolved schedule in safetensors metadata, per-sample
manifests, CSV rows, and `summary.json`. Training remains unchanged until the
inference comparison establishes that late-stage decay is beneficial.
