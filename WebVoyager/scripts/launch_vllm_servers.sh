#!/usr/bin/env bash
set -euo pipefail

WEBVOYAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBVOYAGER_ROOT="${WEBVOYAGER_ROOT:-$(cd "${WEBVOYAGER_SCRIPT_DIR}/.." && pwd)}"
PRO_CUA_ROOT="${PRO_CUA_ROOT:-$(cd "${WEBVOYAGER_ROOT}/.." && pwd)}"

HF_CACHE_DIR="${HF_CACHE_DIR:-${PRO_CUA_ROOT}/.cache/huggingface}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-${PRO_CUA_ROOT}/.cache/vllm}"
PRO_CUA_LIB_DIR="${PRO_CUA_LIB_DIR:-${PRO_CUA_ROOT}/lib}"

mkdir -p "$HF_CACHE_DIR" "$VLLM_DOWNLOAD_DIR"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_DIR}"

if [[ -d "$PRO_CUA_LIB_DIR" ]]; then
  export LIBRARY_PATH="$PRO_CUA_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$PRO_CUA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MODEL=${1:-"Qwen/Qwen3-VL-8B-Instruct"}
BASE_PORT=${BASE_PORT:-8001}

# Use first argument as number of ports/instances; default to 8 if not given
NUM_PORTS=${NUM_PORTS:-8}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Launching $NUM_PORTS vLLM servers for model '$MODEL', starting at port $BASE_PORT..."

for ((i=0; i<NUM_PORTS; i++)); do
  PORT=$((BASE_PORT + i))
  echo "Starting vLLM on GPU $i at port $PORT"
  
  CUDA_VISIBLE_DEVICES=$i VLLM_LOG_LEVEL=warning nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --port "$PORT" \
    --uvicorn-log-level warning \
    --disable-uvicorn-access-log \
    --download_dir "$VLLM_DOWNLOAD_DIR" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-12800}" \
    --limit-mm-per-prompt.image "${VLLM_LIMIT_MM_PER_PROMPT_IMAGE:-5}" \
    > "$LOG_DIR/vllm_gpu${i}.log" 2>&1 &
done

echo "All vLLM servers launched on ports $BASE_PORT to $((BASE_PORT + NUM_PORTS - 1))."

    # --limit-mm-per-prompt image=5 \
