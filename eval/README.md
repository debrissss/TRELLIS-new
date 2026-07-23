# Sparse Structure Evaluation

This directory contains fixed-sample evaluation tools for SS encoder/decoder
fine-tuning runs.

## Prepare A Fixed Dataset

Create a mini TRELLIS dataset root with a subset `metadata.csv` and a `voxels`
symlink back to the source dataset:

```bash
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/prepare_ss_eval_dataset.py \
  --source_root datasets/Facescape/test \
  --output_root datasets/Facescape_ss_eval_test_64 \
  --num_samples 64 \
  --seed 20260718 \
  --min_aesthetic_score 4.5
```

Add `--replace` to regenerate the same output root.

## Run Reconstruction Metrics

```bash
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/evaluate_ss_enc_dec_reconstruction.py \
  --config configs/vae/ss_enc_dec_fine_tune.json \
  --data_root datasets/Facescape_ss_eval_test_64 \
  --checkpoints eval/ss_eval_checkpoints.json \
  --output_dir outputs/eval/ss_enc_dec_eval \
  --batch_size 4
```

Outputs include one per-sample CSV per checkpoint plus `summary.csv` and
`summary.json`.

By default the script uses the posterior mean, matching the training snapshot
path. To mirror stochastic training loss more closely, also run:

```bash
/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/evaluate_ss_enc_dec_reconstruction.py \
  --config configs/vae/ss_enc_dec_fine_tune.json \
  --data_root datasets/Facescape_ss_eval_test_64 \
  --checkpoints eval/ss_eval_checkpoints.json \
  --output_dir outputs/eval/ss_enc_dec_eval_sample_posterior \
  --batch_size 4 \
  --sample_posterior \
  --seed 20260718
```

## Metrics

- `iou`: occupied voxel intersection divided by union.
- `dice_f1`: `2 * intersection / (predicted occupied + GT occupied)`.
- `occupancy_ratio`: predicted occupied voxels divided by GT occupied voxels.
- `gt_occupied_voxels`: GT occupied voxel count.
- `predicted_occupied_voxels`: predicted occupied voxel count.
- `soft_dice_loss`: trainer-style Dice loss computed on sigmoid logits before
  thresholding.
