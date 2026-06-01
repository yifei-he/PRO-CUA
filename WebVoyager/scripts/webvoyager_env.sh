#!/bin/bash

WEBVOYAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBVOYAGER_ROOT="${WEBVOYAGER_ROOT:-$(cd "${WEBVOYAGER_SCRIPT_DIR}/.." && pwd)}"
PRO_CUA_ROOT="${PRO_CUA_ROOT:-$(cd "${WEBVOYAGER_ROOT}/.." && pwd)}"

PRO_CUA_DATA_ROOT="${PRO_CUA_DATA_ROOT:-${PRO_CUA_ROOT}/data}"
PRO_CUA_CKPT_ROOT="${PRO_CUA_CKPT_ROOT:-${PRO_CUA_ROOT}/ckpts}"
LLAMAFACTORY_DATASET_ROOT="${LLAMAFACTORY_DATASET_ROOT:-/tmp/llamafactory_datasets}"

MODEL_API_KEY="${MODEL_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}"
WEBVOYAGER_COLLECT_DATA_SCRIPT="${WEBVOYAGER_COLLECT_DATA_SCRIPT:-${WEBVOYAGER_ROOT}/scripts/webvoyager_collect_data.sh}"
VLLM_BOOT_WAIT_SECONDS="${VLLM_BOOT_WAIT_SECONDS:-120}"

require_llamafactory_root() {
    if [[ -z "${LLAMAFACTORY_ROOT:-}" || ! -d "$LLAMAFACTORY_ROOT" ]]; then
        echo "Set LLAMAFACTORY_ROOT to your LLaMA-Factory checkout before running this script." >&2
        exit 1
    fi
}

bootstrap_round_1_data() {
    local round_1_dir="$1"

    if [[ -d "$round_1_dir" ]] && find "$round_1_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Found existing round_1 data at $round_1_dir; skipping bootstrap collection."
        return
    fi

    if [[ ! -x "$WEBVOYAGER_COLLECT_DATA_SCRIPT" ]]; then
        echo "Missing executable bootstrap data script: $WEBVOYAGER_COLLECT_DATA_SCRIPT" >&2
        exit 1
    fi

    echo "round_1 data not found at $round_1_dir; running $WEBVOYAGER_COLLECT_DATA_SCRIPT"
    ROUND_1_DATA_DIR="$round_1_dir" \
        BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
        MAX_TASKS="${MAX_TASKS:-256}" \
        MAX_ITER="${MAX_STEPS:-20}" \
        TEMPERATURE="${temperature:-0.7}" \
        MODEL_API_KEY="$MODEL_API_KEY" \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        VLLM_BOOT_WAIT_SECONDS="$VLLM_BOOT_WAIT_SECONDS" \
        bash "$WEBVOYAGER_COLLECT_DATA_SCRIPT"
}
