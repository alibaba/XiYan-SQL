
MODEL_NAME_OR_PATH=${1}
EXPR_VERSION=${2}
TEST_SET_PATH=${3}
BATCH_SIZE=${4}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"../train/model/Qwen/Qwen2.5-Coder-0.5B-Instruct"}
EXPR_VERSION=${EXPR_VERSION:-"test_0606"}
TEST_SET_PATH=${TEST_SET_PATH:-"bird_evaluation/eval_set/bird_dev_mschema_0926_short.json"}
BATCH_SIZE=${BATCH_SIZE:-"1"}
CUDA_VISIBLE_DEVICES=0 python sql_infer.py --model_name_or_path ${MODEL_NAME_OR_PATH} --expr_version ${EXPR_VERSION} --test_set_path ${TEST_SET_PATH} --batch_size ${BATCH_SIZE}

# 通过accelerate启动
#CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --config_file config/zero2.yaml sql_infer.py --model_name_or_path ${MODEL_NAME_OR_PATH} --expr_version ${EXPR_VERSION} --test_set_path ${TEST_SET_PATH} --batch_size ${BATCH_SIZE}