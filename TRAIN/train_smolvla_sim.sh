#!/bin/bash
# SmolVLA Training Script (Following LeRobot Methodology)
# Usage: bash train_smolvla_sim.sh

echo "=================================================="
echo "SmolVLA Training (LeRobot Methodology)"
echo "=================================================="
echo ""

# Set environment variables
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLA_Sim/lerobot/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Configuration
CONFIG_FILE="train_config_smolvla_sim.yaml"
NUM_GPUS=5

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# ============================================================
# STEP 1: Verify dataset statistics file exists
# ============================================================
echo "=================================================="
echo "STEP 1: Checking Dataset Statistics"
echo "=================================================="
echo ""

STATS_FILE="dataset_stats_sim.yaml"

if [ ! -f "$STATS_FILE" ]; then
    echo "❌ Error: Dataset statistics file not found: $STATS_FILE"
    echo ""
    echo "You need to compute dataset statistics first!"
    echo "Run:"
    echo "  python compute_dataset_stats.py \\"
    echo "    --dataset_root /path/to/your/hdf5/dataset \\"
    echo "    --output $STATS_FILE"
    echo ""
    exit 1
fi

echo "✅ Found dataset statistics: $STATS_FILE"
echo ""

# ============================================================
# STEP 2: Training
# ============================================================
echo "=================================================="
echo "STEP 2: Training SmolVLA Model"
echo "=================================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting training..."
echo "  Config: $CONFIG_FILE"
echo "  Dataset stats: $STATS_FILE"
echo "  Normalization: MEAN_STD (LeRobot's NormalizerProcessorStep)"
echo ""

# Run training with torchrun (DDP)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train_smolvla_sim.py \
    --config $CONFIG_FILE

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
