#!/bin/bash
set -euo pipefail

WEBVOYAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBVOYAGER_ROOT="${WEBVOYAGER_ROOT:-$(cd "${WEBVOYAGER_SCRIPT_DIR}/.." && pwd)}"
PRO_CUA_ROOT="${PRO_CUA_ROOT:-$(cd "${WEBVOYAGER_ROOT}/.." && pwd)}"
PRO_CUA_DATA_ROOT="${PRO_CUA_DATA_ROOT:-${PRO_CUA_ROOT}/data}"

ROUND_1_DATA_DIR="${ROUND_1_DATA_DIR:-${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/qwen3_8b_512_tasks_20steps_temperature0.7/round_1}"
ROUND_1_OUTPUT_ROOT="$(dirname "$ROUND_1_DATA_DIR")"
ROUND_1_INDEX="$(basename "$ROUND_1_DATA_DIR")"
ROUND_1_INDEX="${ROUND_1_INDEX#round_}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_TASKS="${MAX_TASKS:-256}"
MAX_ITER="${MAX_ITER:-20}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MODEL_API_KEY="${MODEL_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}"

if [[ -d "$ROUND_1_DATA_DIR" ]] && find "$ROUND_1_DATA_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Found existing round_1 data at $ROUND_1_DATA_DIR; skipping collection."
    exit 0
fi

cd "$WEBVOYAGER_ROOT"
mkdir -p logs

pkill -f chrome || true
rm -rf /tmp/.org.chromium.Chromium.*

bash scripts/launch_vllm_servers.sh "$BASE_MODEL"

sleep "${VLLM_BOOT_WAIT_SECONDS:-120}"
nohup python flask_router.py > logs/flask_router.log 2>&1 &

python -u "$WEBVOYAGER_ROOT/run_qwen3_og_parallel_playwright_data_collection_train.py" \
    --test_file "$WEBVOYAGER_ROOT/data/webvoyager_train_data_unique_ids.jsonl" \
    --output_dir "$ROUND_1_OUTPUT_ROOT" \
    --api_key "$MODEL_API_KEY" \
    --max_iter "$MAX_ITER" \
    --max_attached_imgs 1 \
    --temperature "$TEMPERATURE" \
    --window_width 1000 \
    --window_height 1000 \
    --fix_box_color \
    --seed 42 \
    --headless \
    --model qwen3 \
    --model_name "$BASE_MODEL" \
    --num_trials 1 \
    --round "$ROUND_1_INDEX" \
    --max_tasks "$MAX_TASKS"

bash scripts/launch_vllm_servers.sh

sleep "${VLLM_BOOT_WAIT_SECONDS:-120}"
nohup python flask_router.py > logs/flask_router.log 2>&1 &

(
    cd evaluation
    python -u auto_eval_qwen.py \
        --api_key "$OPENAI_API_KEY" \
        --process_dir "$ROUND_1_DATA_DIR" \
        --max_attached_imgs 5 > "$ROUND_1_DATA_DIR/eval.txt"
)

pkill python || true
