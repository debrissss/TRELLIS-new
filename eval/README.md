# TRELLIS Evaluation Tools

`eval/` is organized by evaluation target. Use the normalized entrypoints
below; older duplicated entrypoint scripts have been removed.

## 0. Independent Full-Chain Inference

Normal-conditioned generation is split into four independently runnable
processes. Every process owns its output directory and communicates with the
next process only through `manifest.csv` plus immutable artifacts:

1. `eval.ss_flow_inference`: normal image -> SS latent. It also saves the exact
   preprocessed condition, DINO condition tensor, and post-SS CPU RNG state.
2. `eval.ss_decoder_inference`: SS latent -> occupied SS coordinates. It carries
   the condition and RNG artifact paths into its output manifest.
3. `eval.slat_flow_inference`: occupied SS coordinates -> decoder-ready SLat.
   It restores the saved condition tensor and RNG state, then calls the
   Stable3DGen SLat model, sparse tensor, and sampler implementations.
4. `eval.slat_decoder_inference`: SLat -> mesh or Gaussian PLY.

`eval.full_inference_pipeline` is only an orchestrator. It launches those
processes in order and records their exact commands and terminal statuses in
`run_manifest.json`; it does not contain another copy of model inference.

The reconstruction-only encoder entrypoints remain independent:

- `eval.ss_encoder_inference`: occupancy grid -> SS latent.
- `eval.slat_encoder_inference`: sparse feature grid -> SLat latent.

They are intentionally not inserted into normal-conditioned generation, whose
latents are produced by the two flow stages.

Parity tooling:

- `eval.stable3dgen_reference_inference` runs the unsplit Stable3DGen path with
  the same custom checkpoints and saves intermediate reference artifacts.
- `eval.verify_stable3dgen_parity` compares every handoff, optionally compares a
  second Stable3DGen run, and measures mesh geometry repeatability.

On the current FP16 `spconv` native CUDA path, Stable3DGen's SLat result is not
bitwise repeatable even with identical input, seed, noise, and condition.
Therefore the verifier keeps strict zero-tolerance parity as a failing signal
for SLat/mesh, while separately reporting the Stable3DGen self-repeat scale.
The deterministic prefix through SS coordinates is expected to be bitwise
identical.

## 1. Latent Distribution

Use this to inspect saved SLat latent statistics.

```bash
python -m eval.latent_distribution \
  --data_dir datasets/Facescape/test \
  --latent_model dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16 \
  --output_dir outputs/eval/latent_distribution/pretrained \
  --num_samples 50
```

Outputs:

- `per_sample.csv`
- `summary.json`

Removed old entrypoints:

- `eval/latent_stats.py`
- `eval/analyze_slat_latent_stats.py`

## 2. SLat Encoder + GS Decoder Reconstruction

Single run:

```bash
python -m eval.slat_encoder_gs_decoder_reconstruction single \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --encoder_ckpt outputs/train/<run>/ckpts/encoder_step0001000.pt \
  --decoder_ckpt outputs/train/<run>/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_encoder_gs_decoder_reconstruction/<run> \
  --view_indices 0 \
  --device cuda \
  --fail_on_error
```

Compare existing single-run summaries:

```bash
python -m eval.slat_encoder_gs_decoder_reconstruction many \
  --runs \
    step1000=outputs/eval/slat_encoder_gs_decoder_reconstruction/step1000 \
    step2000=outputs/eval/slat_encoder_gs_decoder_reconstruction/step2000 \
  --output outputs/eval/slat_encoder_gs_decoder_reconstruction/comparison.csv
```

Single-run outputs:

- `metrics.csv`
- `metrics.json`
- `summary.json`
- `failed_samples.json`
- `samples/<sha>_view<idx>/gt.png`
- `samples/<sha>_view<idx>/rec.png`
- `samples/<sha>_view<idx>/diff.png`

Removed old entrypoint:

- `eval/compare_slat_metrics.py`

## 3. Loss Gradient Contribution

This specialized diagnostic is intentionally kept separate.

```bash
python eval/slat_enc_dec_gradient_contrib.py \
  --config configs/vae/slat_enc_dec_gs_fine_tune.json \
  --data_dir datasets/Facescape_eval/slat_gs_eval50 \
  --checkpoints name=1e-6=encoder.pt=decoder.pt \
  --output_dir outputs/eval/gradient_contrib/example \
  --num_samples 8 \
  --view_indices 0
```

Outputs:

- `per_sample.csv`
- `summary.json`
- `failed_samples.json`

## 4. SLat Flow Generation And Evaluation

Generation and evaluation are separate entrypoints:

- `eval.slat_flow_generation` creates flow/mesh artifacts.
- `eval.slat_flow_evaluation` only reads existing artifacts and computes metrics.

Generate fixed samples:

```bash
CUDA_VISIBLE_DEVICES=0 SPCONV_ALGO=native python -m eval.slat_flow_generation flow \
  --config configs/generation/slat_flow_finetune.json \
  --data_dir datasets/Facescape_slat_eval \
  --ckpt outputs/train/<run>/ckpts/denoiser_step0001000.pt \
  --output_dir outputs/eval/slat_flow_generation/<run> \
  --label step1000 \
  --num_samples 16 \
  --seed 20260719 \
  --steps 50 \
  --cfg_strength 3.0 \
  --save_npz \
  --fail_on_error
```

Use `--save_npz` when the generated latent will also be decoded into a triangle
mesh. Flow generation outputs:

- `samples/<sample_id>/cond.png`
- `samples/<sample_id>/generated_grid.png`
- `samples/<sample_id>/gt_grid.png`
- `samples/<sample_id>/generated.ply`
- `samples/<sample_id>/gt.ply`
- `samples/<sample_id>/generated_latent.npz` when `--save_npz` is set
- `manifest.csv`
- `summary.json`

Decode existing generated latents into triangle meshes:

```bash
CUDA_VISIBLE_DEVICES=0 \
SPCONV_ALGO=native \
ATTN_BACKEND=xformers \
SPARSE_ATTN_BACKEND=xformers \
XFORMERS_FORCE_CUTLASS=1 \
python -m eval.slat_flow_generation mesh \
  --runs \
    step1000=outputs/eval/slat_flow_generation/step1000 \
    step2000=outputs/eval/slat_flow_generation/step2000 \
  --mesh_config configs/vae/slat_dec_mesh_fine_tune.json \
  --mesh_decoder_ckpt outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_flow_mesh_generation \
  --require_all_samples
```

When runs live in different latent spaces, pass per-run mesh decoders:

```bash
--run_mesh_decoders \
  step1000=configs/vae/slat_dec_mesh_fine_tune.json=outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt
```

Mesh generation outputs:

- `<run>/meshes/<sample_id>.ply`
- `<run>/manifest.csv`
- `<run>/summary.json`
- `all_runs_summary.csv`
- `summary.json`

Evaluate generated outputs against GT latent decoder renders:

```bash
python -m eval.slat_flow_evaluation gs-image \
  --runs \
    step1000=outputs/eval/slat_flow_generation/step1000 \
    step2000=outputs/eval/slat_flow_generation/step2000 \
  --output_dir outputs/eval/slat_flow_evaluation/gs_image
```

Evaluate existing triangle meshes against GT meshes:

```bash
python -m eval.slat_flow_evaluation mesh \
  --runs \
    step1000=outputs/eval/slat_flow_mesh_generation/step1000 \
    step2000=outputs/eval/slat_flow_mesh_generation/step2000 \
  --data_dir datasets/Facescape/test \
  --output_dir outputs/eval/slat_flow_evaluation/mesh \
  --point_samples 50000 \
  --seed 0 \
  --require_all_samples
```

GS image evaluation outputs:

- `comparison.csv`
- `summary.json`

Mesh evaluation outputs:

- `<run>_per_sample.csv`
- `<run>_summary.json`
- `per_sample.csv`
- `all_runs_summary.csv`
- `summary.json`

The evaluation entrypoint never creates latent, image, or mesh artifacts and
does not load a flow or mesh decoder checkpoint.

Removed old entrypoints:

- `eval/flow_generation.py`
- `eval/compare_flow_generation_metrics.py`

## 5. SLat Mesh Decoder Geometry Reconstruction

Evaluate one checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 SPCONV_ALGO=native python -m eval.slat_mesh_decoder_reconstruction single \
  --config configs/vae/slat_dec_mesh_fine_tune.json \
  --data_dir datasets/Facescape/test \
  --latent_model dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-6_batch8_step0004000 \
  --checkpoints outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_mesh_decoder_reconstruction/slat_dec_mesh_fine_tune \
  --num_samples 50 \
  --point_samples 50000 \
  --seed 0 \
  --require_all_samples
```

Evaluate multiple checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 SPCONV_ALGO=native python -m eval.slat_mesh_decoder_reconstruction many \
  --config configs/vae/slat_dec_mesh_fine_tune.json \
  --data_dir datasets/Facescape/test \
  --latent_model dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-6_batch8_step0004000 \
  --checkpoints \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0000400.pt \
    outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0001000.pt \
  --output_dir outputs/eval/slat_mesh_decoder_reconstruction/slat_dec_mesh_fine_tune \
  --num_samples 50 \
  --point_samples 50000 \
  --seed 0 \
  --require_all_samples
```

Merge existing summary JSON files:

```bash
python -m eval.slat_mesh_decoder_reconstruction compare \
  outputs/eval/slat_mesh_decoder_reconstruction/run/metrics/*_summary.json \
  --output_csv outputs/eval/slat_mesh_decoder_reconstruction/comparison.csv \
  --sort_by chamfer_l1_mean
```

Outputs:

- `eval_samples.json`
- `meshes/<checkpoint>/<sha>.ply`
- `metrics/<checkpoint>_per_sample.csv`
- `metrics/<checkpoint>_summary.json`
- `metrics/all_checkpoints_summary.csv`
- `failures/<checkpoint>_failures.json`

Removed old entrypoints:

- `eval/mesh_decoder_reconstruction.py`
- `eval/compare_mesh_decoder_metrics.py`

## Shared Helpers

Common code lives under `eval/common/`:

- `io.py`: JSON/CSV writing, directory creation, run spec parsing.
- `dataset.py`: metadata reading, fixed indices, path helpers.
- `summary.py`: numeric summaries and summary comparison.
- `image_metrics.py`: image metric exports.
- `mesh_metrics.py`: mesh metric exports.
- `model_loading.py`: mesh decoder loading/export helpers.
- `slat_flow.py`: flow manifest, latent normalization, and decoder spec parsing.

## Metric Direction

- Image `l1`, `mse`, `ssim_loss`, `lpips`, `rec`, `loss`: lower is better.
- Image `psnr`: higher is better.
- Mesh `chamfer_l1`, `chamfer_l2`: lower is better.
- Mesh `fscore_*`, `normal_consistency`: higher is better.
- `kl` is diagnostic; compare it with the training objective and downstream flow behavior.
