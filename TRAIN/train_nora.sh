#!/bin/bash
#
# NORA VLA Training Launch Script
# ================================
#
# This script launches NORA VLA training with custom HDF5 datasets.
#
# Usage:
#   bash train_nora.sh                    # Single GPU training
#   bash train_nora.sh --multi-gpu        # Multi-GPU training
#   bash train_nora.sh --config my.yaml   # Custom config file
#

set -e  # Exit on error

# ============================================================
# Conda Environment Setup (for tmux compatibility)
# ============================================================

# Initialize conda for bash shell (required for tmux)
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    echo "Warning: Could not find conda.sh. Trying to continue anyway..."
fi

# Activate lerobot conda environment
echo "Activating 'lerobot' conda environment..."
conda activate lerobot

# Verify conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "Error: Failed to activate 'lerobot' conda environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✓ Conda environment 'lerobot' activated successfully"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================
# Configuration
# ============================================================

# Default config file
CONFIG_FILE="train_config_nora.yaml"

# Number of GPUs (defaults to 5, can be overridden)
NUM_GPUS=5

# Mixed precision training (set to bf16 for better stability on A100/H100)
MIXED_PRECISION="bf16"

# Multi-GPU training flag (default to true for 5 GPUs)
MULTI_GPU=true

# ============================================================
# Parse Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="--mixed_precision $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash train_nora.sh [--config CONFIG] [--multi-gpu] [--num-gpus N] [--mixed-precision fp16|bf16]"
            exit 1
            ;;
    esac
done

# ============================================================
# Environment Setup
# ============================================================

echo "=========================================="
echo "NORA VLA Training Setup (5-GPU Default)"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Multi-GPU: $MULTI_GPU"
echo "Mixed precision: ${MIXED_PRECISION:-None}"
echo "=========================================="
echo ""
echo "NOTE: Batch size is configured in config file"
echo "      Adjust per_device_batch_size based on VRAM"
echo "      Current: 8 per GPU (80 effective batch)"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if training script exists
if [ ! -f "train_nora.py" ]; then
    echo "Error: Training script not found: train_nora.py"
    exit 1
fi

# Check if dataset adapter exists
if [ ! -f "hdf5_lerobot_adapter.py" ]; then
    echo "Error: Dataset adapter not found: hdf5_lerobot_adapter.py"
    exit 1
fi

# ============================================================
# Launch Training
# ============================================================

echo ""
echo "Starting NORA VLA training..."
echo ""

if [ "$MULTI_GPU" = true ] && [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with Accelerate
    echo "Launching multi-GPU training on $NUM_GPUS GPUs..."

    # Build mixed precision argument
    MP_ARG=""
    if [ -n "$MIXED_PRECISION" ]; then
        MP_ARG="--mixed_precision $MIXED_PRECISION"
    fi

    # Set GPU IDs explicitly
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4

    python -m accelerate.commands.launch \
        --num_processes $NUM_GPUS \
        --num_machines 1 \
        --machine_rank 0 \
        --main_process_port 29500 \
        $MP_ARG \
        train_nora.py \
        --config "$CONFIG_FILE"
else
    # Single GPU training
    echo "Launching single-GPU training..."

    python train_nora.py --config "$CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
