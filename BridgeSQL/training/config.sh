#!/bin/bash
# =============================================================================
# Shared training configuration for BridgeSQL.
# Source this file from SFT and RLVR training scripts.
# =============================================================================

# -----------------------------------------------------------------------------
# Paths (edit these to match your environment)
# -----------------------------------------------------------------------------

# Base model paths (HuggingFace or local)
MODEL_0_5B="Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_1_5B="Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_7B="Qwen/Qwen2.5-Coder-7B-Instruct"

# Training data
SFT_TRAIN_DATA="output/training_data/bridgesql_sft_train.json"
SFT_DEV_DATA="output/training_data/bridgesql_sft_dev.json"
RL_DATA="output/training_data/bridgesql_rl.json"

# Output directories
SFT_OUTPUT_DIR="output/checkpoints/sft"
RLVR_OUTPUT_DIR="output/checkpoints/rlvr"

# Reward plugin
REWARD_PLUGIN="training/reward_utils/plugin.py"

# SQLite execution server (for RLVR reward computation)
export SQLITE_SERVER_URL="${SQLITE_SERVER_URL:-http://localhost:8000}"

# -----------------------------------------------------------------------------
# Hardware
# -----------------------------------------------------------------------------

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC_PER_NODE:-4}"

# Multi-node (for 7B RLVR training: 2 nodes × 4 GPUs)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# -----------------------------------------------------------------------------
# Logging (optional)
# -----------------------------------------------------------------------------

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridgesql}"
