#!/bin/bash
#SBATCH --job-name=glra-rad-mremi
#SBATCH --partition=gpu
#SBATCH --exclude=hpcgpu03,hpcgpu11,hpcgpu14,hpcgpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log
set -euo pipefail
# Run from the package directory:  cd glra_front_rad_package && sbatch run_mremi.sbatch 42
PKG="${SLURM_SUBMIT_DIR:-$PWD}"
DATA="${DATA:-$HOME/mre/data/MRE-MI}"
PYTHON="${PYTHON:-$HOME/mre/venv/bin/python}"
cd "$PKG"
mkdir -p logs runs
export HF_HOME="${HF_HOME:-$HOME/mre/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLRA_N_MAIN=4
export GLRA_N_AUX=8
export GLRA_NUM_WORKERS=4
export NSVCG_BENCH=mremi
SEED="${1:-42}"
"$PYTHON" train_glra_front_rad.py \
  --data_path "$DATA" \
  --image_path "$DATA/images" \
  --aux_image_path "$DATA/images_vg" \
  --aux_image_dict "$DATA/images_vg.pth" \
  --text_model_name bert-base-uncased \
  --image_model_name "$HOME/mre/pretrained_models/ViTL-32-384" \
  --batch_size 16 \
  --epochs 8 \
  --lr 5e-5 \
  --warmup_ratio 0.01 \
  --seed "$SEED" \
  --max_seq_length 128 \
  --num_workers 4
