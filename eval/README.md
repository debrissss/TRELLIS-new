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
  --encoder_ckpt outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt \
  --decoder_ckpt outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_kl1e-7_step1000 \
  --view_indices 0 \
  --device cuda \
  --fail_on_error
```

Evaluate an EMA checkpoint by swapping both checkpoint paths:

```bash
python eval/slat_enc_dec_reconstruction.py \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --encoder_ckpt outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_ema0.9999_step0001000.pt \
  --decoder_ckpt outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_ema0.9999_step0001000.pt \
  --output_dir outputs/eval/slat_kl1e-7_ema_step1000 \
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
outputs/eval/<run>/
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
    kl1e-6_v2=outputs/eval/slat_kl1e-6_v2_step1000 \
    kl1e-7=outputs/eval/slat_kl1e-7_step1000 \
    kl1e-7_ema=outputs/eval/slat_kl1e-7_ema_step1000 \
  --output outputs/eval/slat_checkpoint_comparison.csv
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

## Measure KL Gradient Contribution

Use this diagnostic to estimate how much the weighted KL term contributes to
the local encoder gradient at fixed checkpoints. It keeps the sample set and
view indices fixed across runs, then writes per-sample gradients and summary
statistics.

```bash
CUDA_VISIBLE_DEVICES=0 SPCONV_ALGO=native /root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/slat_enc_dec_gradient_contrib.py \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --checkpoints \
    kl1e-6_step500=1e-6=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6/ckpts/encoder_step0000500.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6/ckpts/decoder_step0000500.pt \
    kl1e-6_step1000=1e-6=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6/ckpts/encoder_step0001000.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6/ckpts/decoder_step0001000.pt \
    kl1e-7_step500=1e-7=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0000500.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_step0000500.pt \
    kl1e-7_step1000=1e-7=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/decoder_step0001000.pt \
    kl1e-8_step500=1e-8=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8/ckpts/encoder_step0000500.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8/ckpts/decoder_step0000500.pt \
    kl1e-8_step1000=1e-8=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8/ckpts/encoder_step0001000.pt=outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_enc_dec_grad_contrib_kl_sweep_step500_1000_eval8_view0 \
  --num_samples 8 \
  --view_indices 0 \
  --sample_posterior \
  --fail_on_error
```

Important output fields:

- `encoder_grad_ratio_kl_total`: `||grad(lambda_kl * kl)|| / ||grad(total)||`.
- `encoder_grad_energy_ratio_kl_total`: squared-norm ratio of the same gradients.
- `encoder_grad_projection_kl_total`: projection of the weighted KL gradient onto the total-gradient direction.
- `encoder_grad_cosine_kl_total`: direction cosine between weighted KL and total gradients.
- `weighted_kl_loss_ratio`: scalar weighted KL contribution to the total loss.

Outputs:

```text
outputs/eval/<run>/
  per_sample.csv
  summary.json
  failed_samples.json
```

## Reliability Notes

- Keep `data_dir`, `metadata.csv`, `selected_sha256.txt`, and `view_indices`
  identical across checkpoint comparisons.
- Prefer deterministic evaluation without `--sample_posterior`. Enable
  `--sample_posterior` only when intentionally measuring stochastic behavior.
- Use `--fail_on_error` for official comparisons so missing/corrupt samples do
  not silently disappear from the mean.
- Training logs are useful for trend monitoring, but fixed eval summaries are
  the safer basis for choosing checkpoint weights.
# FaceScape / SLat Evaluation Tools

## SLat latent distribution

```bash
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/analyze_slat_latent_stats.py \
  --data_dir datasets/Facescape_slat_kl1e-7_nonema_smoke \
  --latent_model dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000 \
  --output_dir outputs/eval/latent_stats_kl1e-7_step1000
```

Outputs:
- `summary.json`
- `per_sample.csv`

## Fixed SLat flow generation

Use `SPCONV_ALGO=native` on the current RTX 5090 environment.

```bash
CUDA_VISIBLE_DEVICES=0 SPCONV_ALGO=native /root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/slat_flow_fixed_generation.py \
  --config configs/generation/slat_flow_finetune_kl1e-7_step1000.json \
  --data_dir datasets/Facescape_slat_kl1e-7_nonema_smoke \
  --ckpt outputs/train/slat_flow_finetune_kl1e-7_step1000/ckpts/denoiser_step0001000.pt \
  --output_dir outputs/eval/slat_flow_kl1e-7_step1000_nonema_fixed_gen \
  --label nonema_step1000 \
  --num_samples 16 \
  --seed 20260719 \
  --steps 50 \
  --cfg_strength 3.0 \
  --fail_on_error
```

Outputs:
- `samples/<sha>/cond.png`
- `samples/<sha>/generated_grid.png`
- `samples/<sha>/gt_grid.png`
- `samples/<sha>/generated.ply`
- `samples/<sha>/gt.ply`
- `manifest.csv`
- `summary.json`

## Flow generation metrics

```bash
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/compare_flow_generation_metrics.py \
  --runs \
    nonema=outputs/eval/slat_flow_kl1e-7_step1000_nonema_fixed_gen \
    ema=outputs/eval/slat_flow_kl1e-7_step1000_ema_fixed_gen \
  --output_dir outputs/eval/slat_flow_kl1e-7_step1000_generation_compare
```

Outputs:
- `comparison.csv`
- `summary.json`

## Mesh decoder reconstruction

The mesh decoder eval uses TRELLIS metadata/latents/checkpoints, but decodes
and exports prediction meshes through Stable3DGen-compatible mesh logic:

- Stable3DGen `SLatMeshDecoder` is used for inference.
- TRELLIS `ElasticSLatMeshDecoder` checkpoints are loaded with `strict=True`.
- Export matches Stable3DGen final inference: `to_trimesh(transform_pose=True)`,
  then +90 degrees around X, then PLY export.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 SPCONV_ALGO=native \
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=flash_attn \
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/mesh_decoder_reconstruction.py \
  --config configs/vae/slat_dec_mesh_fine_tune.json \
  --data_dir datasets/Facescape/test \
  --latent_model dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-6_batch8_step0004000 \
  --checkpoints outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/smoke_mesh_decoder_reconstruction \
  --num_samples 1 \
  --point_samples 100 \
  --seed 0 \
  --require_all_samples
```

Formal comparison example:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 SPCONV_ALGO=native \
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=flash_attn \
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/mesh_decoder_reconstruction.py \
  --config configs/vae/slat_dec_mesh_fine_tune.json \
  --data_dir datasets/Facescape/test \
  --latent_model dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-6_batch8_step0004000 \
  --checkpoints \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0000400.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_ema0.999_step0000400.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0000600.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_ema0.999_step0000600.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0000800.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_ema0.999_step0000800.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_ema0.999_step0001000.pt \
  --output_dir outputs/eval/slat_dec_mesh_fine_tune \
  --num_samples 50 \
  --point_samples 50000 \
  --seed 0 \
  --require_all_samples
```

Outputs:
- `eval_samples.json`: fixed sample list used by every checkpoint.
- `meshes/<checkpoint>/<sha>.ply`: Stable3DGen-aligned predicted mesh exports.
- `metrics/<checkpoint>_per_sample.csv`: per-sample geometry metrics.
- `metrics/<checkpoint>_summary.json`: checkpoint-level metric summary.
- `metrics/all_checkpoints_summary.csv`: merged checkpoint summaries.
- `failures/<checkpoint>_failures.json`: explicit failure records.

Primary geometry metrics:
- `success_rate`: valid predicted mesh rate.
- `chamfer_l1`, `chamfer_l2`: lower is better.
- `fscore_0p005`, `fscore_0p01`, `fscore_0p02`: higher is better.
- `normal_consistency`: higher is better.
- `pred_num_vertices`, `pred_num_faces`, `pred_components`,
  `pred_is_watertight`: structural sanity checks.
