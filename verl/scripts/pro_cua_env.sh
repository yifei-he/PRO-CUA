#!/bin/bash

# Shared defaults for PRO-CUA training entry points.
PRO_CUA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRO_CUA_ROOT="${PRO_CUA_ROOT:-$(cd "${PRO_CUA_SCRIPT_DIR}/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-${PRO_CUA_ROOT}/verl}"
WEBVOYAGER_ROOT="${WEBVOYAGER_ROOT:-${PRO_CUA_ROOT}/WebVoyager}"
WEBVOYAGER_COLLECT_DATA_SCRIPT="${WEBVOYAGER_COLLECT_DATA_SCRIPT:-${WEBVOYAGER_ROOT}/scripts/webvoyager_collect_data.sh}"

PRO_CUA_DATA_ROOT="${PRO_CUA_DATA_ROOT:-${PRO_CUA_ROOT}/data}"
PRO_CUA_CKPT_ROOT="${PRO_CUA_CKPT_ROOT:-${PRO_CUA_ROOT}/ckpts}"
RAY_TMPDIR="${RAY_TMPDIR:-${PRO_CUA_ROOT}/tmp/ray}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${PRO_CUA_ROOT}/tmp/ray_spill}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export MODEL_API_KEY="${MODEL_API_KEY:-}"

bootstrap_round_1_data() {
    local round_1_dir="$1"

    if [[ -d "$round_1_dir" ]] && find "$round_1_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Found existing round_1 data at $round_1_dir; skipping bootstrap collection."
        return
    fi

    if [[ ! -x "$WEBVOYAGER_COLLECT_DATA_SCRIPT" ]]; then
        echo "Missing executable bootstrap data script: $WEBVOYAGER_COLLECT_DATA_SCRIPT" >&2
        echo "Set WEBVOYAGER_COLLECT_DATA_SCRIPT to the data-collection script path, or create round_1 data at $round_1_dir." >&2
        exit 1
    fi

    echo "round_1 data not found at $round_1_dir; running $WEBVOYAGER_COLLECT_DATA_SCRIPT"
    (
        cd "$WEBVOYAGER_ROOT"
        ROUND_1_DATA_DIR="$round_1_dir" \
            BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
            MAX_TASKS="${MAX_TASKS:-256}" \
            MODEL_API_KEY="$MODEL_API_KEY" \
            OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}" \
            bash "$WEBVOYAGER_COLLECT_DATA_SCRIPT"
    )

    if [[ ! -d "$round_1_dir" ]] || ! find "$round_1_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Bootstrap script completed, but round_1 data is still missing at $round_1_dir." >&2
        exit 1
    fi
}

mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR"
export RAY_TMPDIR
export RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${RAY_SPILL_DIR}\"}}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

if [[ -n "$HF_TOKEN" ]]; then
    hf auth login --token "$HF_TOKEN"
else
    echo "HF_TOKEN is not set; skipping Hugging Face login."
fi
