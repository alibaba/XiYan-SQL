#!/bin/bash
# Set NCCL_IB as needed

# wandb has been deprecated
#export WANDB_PROJECT='xxx'

# Basic distributed training configuration
GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-12547}
#CONFIG="config/zero3.yaml"
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
        --per_device_train_batch_size $BZ \
        --load_best_model_at_end False\
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $ACC_STEP \
        --save_strategy "steps" \
        --eval_strategy "no" \
        --eval_steps $SAVE_STEP \
        --save_steps $SAVE_STEP \
        --save_total_limit 50 \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --adam_beta2 0.95 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --log_level "debug" \
        --logging_steps 10 \
        --report_to "none" \
        --model_max_length $MAX_LENGTH \
        --gradient_checkpointing True \
        --predict_with_generate True \
        --include_inputs_for_metrics True \
        --use_moe_lora $USE_MOE_LORA \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $((LORA_R * LORA_SCALE)) \
        --torch_compile True \
        --group_by_length $GROUP_BY_LENGTH \
        --use_flash_attention True \
        --model_type $MODEL_TYPE \
        --num_experts $num_experts \
        --lora_route_type $lora_route_type \
        --moe_lora_target_modules "${moe_lora_target_modules[@]}" \
        --output_router_logits $output_router_logits \
        --enable_dialect_router $enable_dialect_router \
        --dialect_router_loss_coef $dialect_router_loss_coef \
        --num_experts_per_tok $num_experts_per_tok \
        --dialect_num $dialect_num \
        --enable_label_smooth $enable_label_smooth \
        --smooth_factor $smooth_factor \
        --share_expert_num $share_expert_num \
        --train_dialects_num_map "$train_dialects_num_map" \
        --eval_dialects_num_map "$eval_dialects_num_map" \
        --dataloader_num_workers 8 \
        --hard_dialect_router $hard_dialect_router \
        --use_in_group_balance $use_in_group_balance \
        --seed 42 \
        --bf16 \
        --expr_id $EXPR_ID
#        --eval_data_path $EVAL_DATA \
}


EXPR_ID="momq_configxxx_date_xxx"

MODEL="model/Qwen/Qwen2___5-Coder-7B-Instruct"
MODEL_TYPE="momq-qwen"


# training config

BZ=1
EPOCH=5
LR=1e-5
WEIGHT_DECAY=0.1
ACC_STEP=1
MAX_LENGTH=8192
SAVE_STEP=200


# momq moe config

## Use Lora as a Moe expert. No need to turn on Lora mode
USE_LORA=False
USE_MOE_LORA=True
lora_route_type='token'
moe_lora_target_modules=("down_proj")
# moe_lora_target_modules=("q_proj" "k_proj" "v_proj" "o_proj" "down_proj")
enable_dialect_router=True
output_router_logits=True
router_aux_loss_coef=0.001
dialect_router_loss_coef=0.01
hard_dialect_router=False
use_in_group_balance=False
enable_label_smooth=False
smooth_factor=0.01
GROUP_BY_LENGTH=True


LORA_R=128
LORA_SCALE=2
num_experts=24
num_experts_per_tok=2
share_expert_num=2
# mysql pg sqlite
dialect_num=3


# You can disable the evaluation during training
#EVAL_DATA="/path/to/data/xxx.json"
DATA="/path/to/data/xxx.json"
OUTPUT="/path/to/output/momq/${EXPR_ID}/"
run_training $DATA $OUTPUT

