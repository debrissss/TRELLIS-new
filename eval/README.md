# TRELLIS Evaluation Tools

This directory contains fixed-data evaluation utilities for comparing TRELLIS
SLat encoder/decoder checkpoints. The goal is to compare checkpoints on the
same samples and camera views, instead of relying only on stochastic training
logs.

## Prepare a Fixed FaceScape Eval Subset

Create a small eval dataset from `datasets/Facescape/test`:

```bash
python eval/prepare_facescape_eval_subset.py \
  --source_dir datasets/Facescape/test \
  --output_dir datasets/Facescape_eval/slat_gs_eval50 \
  --num_samples 50 \
  --seed 20260718 \
  --feature_model dinov2_vitl14_reg
```

By default the script creates symlinks to save space. Use `--copy` when the
eval subset needs to be moved to another machine.

The output layout is:

```text
datasets/Facescape_eval/slat_gs_eval50/
  metadata.csv
  selected_sha256.txt
  manifest.json
  renders/<sha>/
  features/dinov2_vitl14_reg/<sha>.npz
```

The script filters out samples with missing `renders/<sha>/transforms.json`,
missing feature files, unreadable feature files, or a false
`feature_dinov2_vitl14_reg` metadata flag.

## Evaluate One Checkpoint Pair

Evaluate a non-EMA checkpoint:

```bash
python eval/slat_enc_dec_reconstruction.py \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --encoder_ckpt outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt \
  --decoder_ckpt outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_step0001000.pt \
  --output_dir eval_outputs/slat_kl1e-7_step1000 \
  --view_indices 0 \
  --device cuda \
  --fail_on_error
```

Evaluate an EMA checkpoint by swapping both checkpoint paths:

```bash
python eval/slat_enc_dec_reconstruction.py \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --encoder_ckpt outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_ema0.9999_step0001000.pt \
  --decoder_ckpt outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_ema0.9999_step0001000.pt \
  --output_dir eval_outputs/slat_kl1e-7_ema_step1000 \
  --view_indices 0 \
  --device cuda \
  --fail_on_error
```

Use comma-separated views for a more robust but slower multi-view score:

```bash
--view_indices 0,4,8,12
```

Outputs:

```text
eval_outputs/<run>/
  metrics.csv
  metrics.json
  summary.json
  failed_samples.json
  samples/<sha>_view<idx>/gt.png
  samples/<sha>_view<idx>/rec.png
  samples/<sha>_view<idx>/diff.png
```

## Compare Evaluated Runs

```bash
python eval/compare_slat_metrics.py \
  --runs \
    kl1e-6_v2=eval_outputs/slat_kl1e-6_v2_step1000 \
    kl1e-7=eval_outputs/slat_kl1e-7_step1000 \
    kl1e-7_ema=eval_outputs/slat_kl1e-7_ema_step1000 \
  --output eval_outputs/slat_checkpoint_comparison.csv
```

## Metric Direction

- `l1`: lower is better.
- `mse`: lower is better.
- `psnr`: higher is better.
- `ssim_loss`: lower is better; this is `1 - SSIM`.
- `lpips`: lower is better.
- `rec`: lower is better; this is `l1 + lambda_ssim * ssim_loss + lambda_lpips * lpips`.
- `loss`: lower is better for the eval objective; this is `rec + lambda_kl * kl`.
- `kl`: diagnostic only. Very low or very high can both be suspicious depending on the downstream flow model.

## Reliability Notes

- Keep `data_dir`, `metadata.csv`, `selected_sha256.txt`, and `view_indices`
  identical across checkpoint comparisons.
- Prefer deterministic evaluation without `--sample_posterior`. Enable
  `--sample_posterior` only when intentionally measuring stochastic behavior.
- Use `--fail_on_error` for official comparisons so missing/corrupt samples do
  not silently disappear from the mean.
- Training logs are useful for trend monitoring, but fixed eval summaries are
  the safer basis for choosing checkpoint weights.
