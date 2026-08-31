#!/bin/bash
# =============================================================================
# BridgeSQL RLVR (GRPO) Training Script
#
# Prerequisites:
#   - SFT checkpoint available (from training/sft/train.sh)
#   - SQLite execution server running (training/reward_utils/sqlite_server.py)
#
# Usage:
#   MODEL_SIZE=0.5b SFT_CKPT=output/checkpoints/sft/0.5b/checkpoint-400 \
#       bash training/rlvr/train.sh
#
#   MODEL_SIZE=7b SFT_CKPT=output/checkpoints/sft/7b/checkpoint-280 \
#       bash training/rlvr/train.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

MODEL_SIZE="${MODEL_SIZE:-7b}"

if [ -z "$SFT_CKPT" ]; then
    echo "ERROR: SFT_CKPT must be set to the SFT checkpoint path."
    echo "  e.g., SFT_CKPT=output/checkpoints/sft/7b/checkpoint-280"
    exit 1
fi

# -----------------------------------------------------------------------------
# Model-specific hyperparameters
# -----------------------------------------------------------------------------

case "$MODEL_SIZE" in
    0.5b|0.5B)
        LEARNING_RATE=5e-6
        BATCH_SIZE=4
        GRAD_ACCUM=4
        MAX_COMPLETION=2048
        # Effective batch: 4 × 4 GPUs × 1 node × 4 accum = 64 prompts
        ;;
    1.5b|1.5B)
        LEARNING_RATE=5e-6
        BATCH_SIZE=4
        GRAD_ACCUM=4
        MAX_COMPLETION=2048
        # Effective batch: 4 × 4 GPUs × 1 node × 4 accum = 64 prompts
        ;;
    7b|7B)
        LEARNING_RATE=1e-6
        BATCH_SIZE=2
        GRAD_ACCUM=4
        MAX_COMPLETION=2048
        # Effective batch: 2 × 4 GPUs × 2 nodes × 4 accum = 64 prompts
        # Requires NNODES=2 (see README for multi-node setup)
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE (choose from: 0.5b, 1.5b, 7b)"
        exit 1
        ;;
esac

OUTPUT="${RLVR_OUTPUT_DIR}/${MODEL_SIZE}"

echo "============================================="
echo " BridgeSQL RLVR (GRPO) Training"
echo " SFT Checkpoint: ${SFT_CKPT}"
echo " Size:  ${MODEL_SIZE}"
echo " LR:    ${LEARNING_RATE}"
echo " Nodes: ${NNODES} (rank ${NODE_RANK})"
echo " Server: ${SQLITE_SERVER_URL}"
echo " Output: ${OUTPUT}"
echo "============================================="

# -----------------------------------------------------------------------------
# Launch training
# -----------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC \
swift rlhf \
    --rlhf_type grpo \
    --model "$SFT_CKPT" \
    --reward_funcs sql_acc \
    --external_plugins "$REWARD_PLUGIN" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 30000 \
    --max_length 25000 \
    --truncation_strategy left \
    --sleep_level 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "$RL_DATA" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length $MAX_COMPLETION \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --beta 0.001 \
    --dynamic_sample true \
    --max_resample_times 3 \
    --num_iterations 1 \
    --deepspeed zero3 \
    --report_to wandb \
    --log_completions true \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 50 \
    --output_dir "$OUTPUT"
