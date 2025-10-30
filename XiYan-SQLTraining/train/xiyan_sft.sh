#!/bin/bash
# Set NCCL_IB as needed

# wandb has been deprecated, now use swanlab
#export WANDB_PROJECT='xxx'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Basic distributed training configuration
GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-12547}
#DS_CONFIG="config/zero3.yaml"
DS_CONFIG="config/zero2.yaml"



run_training() {
    local DATA=$1
    local OUTPUT=$2
    accelerate launch --config_file $DS_CONFIG --num_machines $NNODES --num_processes $WORLD_SIZE --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    sft4xiyan.py \
        --save_only_model True \
        --resume False \
        --model_name_or_path $MODEL \
        --data_path $DATA \
        --output_dir $OUTPUT \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BATCH_SIZE \
        --load_best_model_at_end False\
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $ACC_STEP \
        --save_strategy "steps" \
        --eval_strategy "no" \
        --eval_steps $SAVE_STEP \
        --save_steps $SAVE_STEP \
        --save_total_limit 100 \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --adam_beta2 0.95 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --log_level "debug" \
        --logging_steps 1 \
        --report_to "none" \
        --model_max_length $MAX_LENGTH \
        --lazy_preprocess False \
        --gradient_checkpointing True \
        --predict_with_generate True \
        --include_inputs_for_metrics True \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $((LORA_R * LORA_SCALE)) \
        --do_shuffle $SHUFFLE \
        --torch_compile False \
        --group_by_length $GROUP_BY_LENGTH \
        --model_type "auto" \
        --use_flash_attention True \
        --bf16 \
        --expr_id $EXPR_ID
        # --eval_data_path $EVAL_DATA
}


# Try to conduct an experiment
EXPR_ID="namexxx_configxxx_date_xxx"

MODEL="model/Qwen/Qwen2___5-Coder-7B-Instruct"
#MODEL="model/Qwen/Qwen2.5-Coder-0.5B-Instruct"
EPOCH=5
LR=1e-6
WEIGHT_DECAY=0.1
MAX_LENGTH=10240

USE_LORA=False
LORA_R=512
LORA_SCALE=1

BATCH_SIZE=1
ACC_STEP=2
SAVE_STEP=500
# open group by length
GROUP_BY_LENGTH=True

# You can disable the evaluation during training
#EVAL_DATA="/path/to/data/xxx.json"
#DATA="/path/to/data/xxx.json"
#OUTPUT="/path/to/output/dense/${EXPR_ID}/"


DATA="datasets/train_examples.json"
OUTPUT="output/dense/${EXPR_ID}/"

run_training $DATA $OUTPUT











