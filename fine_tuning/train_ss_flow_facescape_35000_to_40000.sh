#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535
echo "[INFO] Open file limit: $(ulimit -n)"

CONDA_ENV="${CONDA_ENV:-trellis}"
CONFIG="${CONFIG:-configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape_35000_to_40000.json}"
DATA_DIR="${DATA_DIR:-datasets/Facescape/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ss_flow_facescape_finetune_35000_to_40000}"
NUM_GPUS="${NUM_GPUS:-1}"
CKPT="${CKPT:-none}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python train.py \
    --config "${CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    --ckpt "${CKPT}" \
    "$@"
