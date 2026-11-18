#!/bin/bash
#SBATCH --job-name=kathavaani-studio
#SBATCH --output=logs/serve_%j.log
#SBATCH --error=logs/serve_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HOME=/var/tmp/dgx-s-bmu-soet-230737/hf_cache
export TORCH_HOME=/var/tmp/dgx-s-bmu-soet-230737/torch_cache
export PATH="/var/tmp/dgx-s-bmu-soet-230737/bin:/var/tmp/dgx-s-bmu-soet-230737/venv/bin:$PATH"

NODE_IP=$(hostname -I | awk '{print $1}')
echo "=========================================================="
echo " KathaVaani Studio Server starting on node $(hostname) ($NODE_IP)"
echo " Port forward from your local laptop using:"
echo "   ssh -L 8000:$NODE_IP:8000 $USER@<headnode-ip>"
echo " Then open: http://localhost:8000"
echo "=========================================================="

python3 -m server.app --host 0.0.0.0 --port 8000
