#!/bin/bash
#SBATCH --job-name=kathavaani-train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -euo pipefail

export HF_HOME=/var/tmp/dgx-s-bmu-soet-230737/hf_cache
export TORCH_HOME=/var/tmp/dgx-s-bmu-soet-230737/torch_cache
export PATH="/var/tmp/dgx-s-bmu-soet-230737/bin:/var/tmp/dgx-s-bmu-soet-230737/venv/bin:$PATH"

echo "=== Starting Training Job on $(hostname) at $(date) ==="
nvidia-smi

# Resumable training run
python3 -m kathavaani.tts.finetune.train \
    --config configs/train_xtts.yaml \
    --resume \
    --output_dir checkpoints/xtts_hindi_expressive

echo "=== Training Finished at $(date) ==="
