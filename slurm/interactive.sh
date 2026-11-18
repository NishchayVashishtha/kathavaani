#!/bin/bash
# KathaVaani Interactive GPU Session (SLURM template)
# Usage: ./slurm/interactive.sh

srun \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=24G \
    --time=04:00:00 \
    --pty bash
