#!/bin/bash
# =============================================================================
# BridgeSQL Evaluation Script
#
# Evaluate a model on all seven cross-domain Text-to-SQL benchmarks.
#
# Prerequisites:
#   - SQLite execution server running (training/reward_utils/sqlite_server.py)
#   - Evaluation datasets prepared in eval/ (Step 3.1-3.3)
#
# Usage:
#   bash evaluation/run_eval.sh --model_path <checkpoint_path>
#   bash evaluation/run_eval.sh --model_path <path> --datasets spider_test bird_dev
#   bash evaluation/run_eval.sh --model_path <path> --tp 2
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

EVAL_DIR="${EVAL_DIR:-eval/}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval_results/}"
SERVER_URL="${SQLITE_SERVER_URL:-http://localhost:8000}"
TP_SIZE="${TP_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.01}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_WORKERS="${MAX_WORKERS:-20}"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------

MODEL_PATH=""
DATASETS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"; shift 2 ;;
        --eval_dir)
            EVAL_DIR="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --server_url)
            SERVER_URL="$2"; shift 2 ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! $1 == --* ]]; do
                DATASETS="$DATASETS $1"; shift
            done
            ;;
        --tp)
            TP_SIZE="$2"; shift 2 ;;
        --temperature)
            TEMPERATURE="$2"; shift 2 ;;
        --max_tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash evaluation/run_eval.sh --model_path <checkpoint_path>"
    echo ""
    echo "Options:"
    echo "  --model_path    Path to model or checkpoint (required)"
    echo "  --eval_dir      Evaluation dataset directory (default: eval/)"
    echo "  --output_dir    Output directory (default: output/eval_results/)"
    echo "  --server_url    SQLite server URL (default: \$SQLITE_SERVER_URL)"
    echo "  --datasets      Specific datasets (default: all 7 benchmarks)"
    echo "  --tp            Tensor parallel size (default: 4)"
    echo "  --temperature   Sampling temperature (default: 0.01)"
    echo "  --max_tokens    Max generation tokens (default: 2048)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------

echo "Checking SQLite server at ${SERVER_URL}..."
if ! curl -s --max-time 5 "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "[ERROR] SQLite server not reachable at ${SERVER_URL}"
    echo "Start it with: python training/reward_utils/sqlite_server.py --db_dir databases/"
    exit 1
fi
echo "Server OK."

# -----------------------------------------------------------------------------
# Run evaluation
# -----------------------------------------------------------------------------

CMD="python evaluation/evaluate.py \
    --model_path $MODEL_PATH \
    --eval_dir $EVAL_DIR \
    --output_dir $OUTPUT_DIR \
    --server_url $SERVER_URL \
    --tensor_parallel_size $TP_SIZE \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --max_workers $MAX_WORKERS"

if [ -n "$DATASETS" ]; then
    CMD="$CMD --datasets $DATASETS"
fi

echo "============================================="
echo " BridgeSQL Evaluation"
echo " Model:   ${MODEL_PATH}"
echo " Datasets: ${DATASETS:-all}"
echo " Server:  ${SERVER_URL}"
echo " Output:  ${OUTPUT_DIR}"
echo "============================================="

eval $CMD
