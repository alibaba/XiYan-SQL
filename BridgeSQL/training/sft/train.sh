#!/bin/bash
# =============================================================================
# BridgeSQL SFT Training Script
#
# Usage:
#   MODEL_SIZE=0.5b bash training/sft/train.sh
#   MODEL_SIZE=1.5b bash training/sft/train.sh
#   MODEL_SIZE=7b   bash training/sft/train.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

MODEL_SIZE="${MODEL_SIZE:-7b}"

# -----------------------------------------------------------------------------
# Model-specific hyperparameters
# -----------------------------------------------------------------------------

case "$MODEL_SIZE" in
    0.5b|0.5B)
        MODEL_PATH="$MODEL_0_5B"
        LEARNING_RATE=3e-5
        BATCH_SIZE=8
        EVAL_BATCH_SIZE=8
        GRAD_ACCUM=8
        ;;
    1.5b|1.5B)
        MODEL_PATH="$MODEL_1_5B"
        LEARNING_RATE=3e-5
        BATCH_SIZE=8
        EVAL_BATCH_SIZE=8
        GRAD_ACCUM=8
        ;;
    7b|7B)
        MODEL_PATH="$MODEL_7B"
        LEARNING_RATE=2e-5
        BATCH_SIZE=4
        EVAL_BATCH_SIZE=4
        GRAD_ACCUM=16
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE (choose from: 0.5b, 1.5b, 7b)"
        exit 1
        ;;
esac

OUTPUT="${SFT_OUTPUT_DIR}/${MODEL_SIZE}"

echo "============================================="
echo " BridgeSQL SFT Training"
echo " Model: ${MODEL_PATH}"
echo " Size:  ${MODEL_SIZE}"
echo " LR:    ${LEARNING_RATE}"
echo " Output: ${OUTPUT}"
echo "============================================="

# -----------------------------------------------------------------------------
# Launch training
# -----------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
NPROC_PER_NODE=$NPROC \
swift sft \
    --train_type full \
    --torch_dtype bfloat16 \
    --model "$MODEL_PATH" \
    --dataset "$SFT_TRAIN_DATA" \
    --val_dataset "$SFT_DEV_DATA" \
    --dataset_num_proc 8 \
    --max_length 32768 \
    --padding_free true \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --attn_impl flash_attention_2 \
    --deepspeed zero3 \
    --report_to wandb \
    --logging_steps 5 \
    --num_train_epochs 3 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 100 \
    --eval_strategy steps \
    --eval_steps 20 \
    --output_dir "$OUTPUT"
