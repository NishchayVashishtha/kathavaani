#!/bin/bash
#SBATCH --job-name=kathavaani-infer
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

export HF_HOME=/var/tmp/dgx-s-bmu-soet-230737/hf_cache
export TORCH_HOME=/var/tmp/dgx-s-bmu-soet-230737/torch_cache
export PATH="/var/tmp/dgx-s-bmu-soet-230737/bin:/var/tmp/dgx-s-bmu-soet-230737/venv/bin:$PATH"

python3 -m kathavaani.cli "$@"
