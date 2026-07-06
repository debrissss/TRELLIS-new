#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535
CONDA_ENV="${CONDA_ENV:-trellis5090}"
CONDA_BASE="${CONDA_BASE:-/root/autodl-tmp/mamba_envs}"
PYTHON="${PYTHON:-/root/autodl-tmp/mamba_envs/trellis5090/bin/python}"
if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS}" = "0" ]; then
  export OMP_NUM_THREADS=8
fi
if [ -z "${MKL_NUM_THREADS:-}" ] || [ "${MKL_NUM_THREADS}" = "0" ]; then
  export MKL_NUM_THREADS=8
fi

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cd /root/autodl-tmp/TRELLIS-new
"${PYTHON}" train.py \
  --config /root/autodl-tmp/TRELLIS-new/configs/generation/overfit/ss_flow_facescape_overfit_1.json \
  --data_dir /root/autodl-tmp/TRELLIS-new/datasets/Facescape/overfit_1 \
  --output_dir /root/autodl-tmp/TRELLIS-new/outputs/ss_flow_facescape_overfit_1 \
  --num_gpus "${NUM_GPUS:-1}" \
  --ckpt none \
  "$@"
