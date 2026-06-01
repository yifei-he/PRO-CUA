#!/bin/bash
# start the vllm servers
# rejection sampling with llamafactory
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/webvoyager_env.sh"
export PATH="${CONDA_PREFIX:-}/bin:$PATH"

TOTAL_ROUNDS=10
MIN_LR=1e-7
BASE_LR=5e-6
MAX_TASKS=256
MODEL_NAME=qwen3_8b
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
EXP_NAME=sft
DATA_MODE=correct_traj
MAX_STEPS=20
temperature=0.7
n_epochs=2
FULL_NAME=${EXP_NAME}_${MODEL_NAME}_${MAX_TASKS}tasks_${MAX_STEPS}steps_${DATA_MODE}_temperature${temperature}_${n_epochs}epochs_${BASE_LR}
CKPT_ROOT=${PRO_CUA_CKPT_ROOT}/qwen3-vl-8b

for ((round=1; round<=TOTAL_ROUNDS; round++)); do

echo "Processing round $round"

conda activate verl-agent
cd "$WEBVOYAGER_ROOT"
if [ "$round" -eq 1 ]; then
  main_folder=${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/qwen3_8b_512_tasks_${MAX_STEPS}steps_temperature${temperature}/round_1
  bootstrap_round_1_data "$main_folder"
else
  main_folder=${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}/round_$round
fi
python construct_dataset.py --algo $FULL_NAME --correct_traj_only --round $round --main_folder $main_folder --max_tasks ${MAX_TASKS}

dataset_name=${FULL_NAME}_round_${round}
dataset_file=${main_folder}/${FULL_NAME}_round_${round}.json
dataset_dir=${LLAMAFACTORY_DATASET_ROOT}/${FULL_NAME}/round_${round}
python scripts/generate_lf_dataset_config.py \
  --dataset-name "$dataset_name" \
  --dataset-file "$dataset_file" \
  --output-dir "$dataset_dir"

conda deactivate

conda activate llamafactory

require_llamafactory_root
cd "$LLAMAFACTORY_ROOT"

if [ "$round" -eq 1 ]; then
  model_name_or_path=$BASE_MODEL
else
  model_name_or_path=${CKPT_ROOT}/${FULL_NAME}/round_$((round-1))
fi

# CURRENT_LR=$(python3 -c "
# import math
# lr_max = float('${BASE_LR}')
# lr_min = float('${MIN_LR}')
# round_idx = int('${round}') - 1
# total = int('${TOTAL_ROUNDS}')
# print(f'{lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(round_idx * math.pi / total)):.8f}')
# ")    
CURRENT_LR=${BASE_LR}
echo "Using Learning Rate: $CURRENT_LR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
llamafactory-cli train examples/train_full/qwen3vl_full_sft.yaml \
  model_name_or_path=$model_name_or_path \
  dataset=$dataset_name \
  dataset_dir=$dataset_dir \
  output_dir=${CKPT_ROOT}/${FULL_NAME}/round_${round} \
  run_name=${FULL_NAME}_round_${round} \
  learning_rate=$CURRENT_LR \
  per_device_train_batch_size=2 \
  gradient_accumulation_steps=4 \
  warmup_steps=1 \
  num_train_epochs=${n_epochs} \

conda deactivate

conda activate verl-agent

cd "$WEBVOYAGER_ROOT"
export PATH="${CONDA_PREFIX:-}/bin:$PATH"

pkill -f chrome || true
rm -rf /tmp/.org.chromium.Chromium.*
rm -rf "${CKPT_ROOT}/${FULL_NAME}/round_$round"/checkpoint-*

model_path=${CKPT_ROOT}/${FULL_NAME}/round_$round

bash scripts/launch_vllm_servers.sh "$model_path"

sleep "$VLLM_BOOT_WAIT_SECONDS"
nohup python flask_router.py > logs/flask_router.log 2>&1 &

########################################################
### evaluation
########################################################
pkill -f chrome || true
rm -rf /tmp/.org.chromium.Chromium.*

start=`date +%s`
python -u "$WEBVOYAGER_ROOT/run_qwen3_og_parallel_playwright_data_collection_train.py" \
    --test_file "$WEBVOYAGER_ROOT/data/WebVoyager_data_clean.jsonl" \
    --output_dir "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_data_clean/${FULL_NAME}" \
    --api_key "$MODEL_API_KEY" \
    --max_iter 30 \
    --max_attached_imgs 1 \
    --temperature 0 \
    --window_width 1000 \
    --window_height 1000 \
    --fix_box_color \
    --seed 42 \
    --headless \
    --model qwen3 \
    --model_name $model_path \
    --num_trials 1 \
    --round $((round+1)) \

end=`date +%s`

runtime=$((end-start))
echo "Script executed in $runtime seconds."

########################################################
### rollout for 512 trajectories
########################################################
if [ "$round" -ne "$TOTAL_ROUNDS" ]; then
    start=`date +%s`
    python -u "$WEBVOYAGER_ROOT/run_qwen3_og_parallel_playwright_data_collection_train.py" \
        --test_file "$WEBVOYAGER_ROOT/data/webvoyager_train_data_unique_ids.jsonl" \
        --output_dir "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}" \
        --api_key "$MODEL_API_KEY" \
        --max_iter 20 \
        --max_attached_imgs 1 \
        --temperature 1 \
        --window_width 1000 \
        --window_height 1000 \
        --fix_box_color \
        --seed 42 \
        --headless \
        --model qwen3 \
        --model_name $model_path \
        --num_trials 1 \
        --round $((round+1)) \
        --save_log_prob \
        --max_tasks ${MAX_TASKS}
    end=`date +%s`
    runtime=$((end-start))
    echo "Script executed in $runtime seconds."

    pkill python || true

    ########################################################
    ### delete the task folders with non-consecutive screenshots
    ########################################################
    rollout_root=${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}/round_$((round+1))
    find "$rollout_root" -type f -name 'screenshot*.png' -printf '%h\n' | sort -u | while read -r task_dir; do
        has_gap=0
        prev_num=""

        while read -r screenshot_path; do
            filename=$(basename "$screenshot_path")
            screenshot_num=${filename#screenshot}
            screenshot_num=${screenshot_num%.png}

            if [ -n "$prev_num" ] && [ "$screenshot_num" -ne $((prev_num + 1)) ]; then
                has_gap=1
                break
            fi

            prev_num=$screenshot_num
        done < <(find "$task_dir" -maxdepth 1 -type f -name 'screenshot*.png' | sort -V)

        if [ "$has_gap" -eq 1 ]; then
            echo "Deleting task folder with non-consecutive screenshots: $task_dir"
            rm -rf "$task_dir"
        fi
    done
fi

########################################################
### start the evaluation
########################################################
bash scripts/launch_vllm_servers.sh

sleep "$VLLM_BOOT_WAIT_SECONDS"
nohup python flask_router.py > logs/flask_router.log 2>&1 &

########################################################
### label the trajectories
########################################################
cd evaluation
DIRS=(
    "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_data_clean/${FULL_NAME}/round_$((round+1))"
    "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}/round_$((round+1))"
)

# Loop through each directory
for DIR in "${DIRS[@]}"; do
    echo "Processing directory: $DIR"
    python -u auto_eval_qwen.py \
        --api_key "$OPENAI_API_KEY" \
        --process_dir "$DIR" \
        --max_attached_imgs 5 > "$DIR/eval.txt"
    echo "Completed: $DIR"
    echo "----------------------------------------"
done

echo "All evaluations completed!"
cd ..
pkill python || true

done
