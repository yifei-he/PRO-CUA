#!/bin/bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pro_cua_env.sh"
export PATH="$CONDA_PREFIX/bin:$PATH"
mkdir -p "$HOME/lib"
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/lib/libcuda.so"
export LIBRARY_PATH="$HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TOTAL_ROUNDS=10
MIN_LR=1e-7
BASE_LR=5e-6
CHUNK_SIZE=64
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
EXP_NAME=step_grpo
MODEL_NAME=qwen3_8b
MAX_TASKS=256
FULL_NAME=${EXP_NAME}_${MODEL_NAME}_${MAX_TASKS}tasks_correct_traj_verl_rule_${BASE_LR}

for ((round=1; round<=TOTAL_ROUNDS; round++)); do

#######################################################
## construct dataset
#######################################################
if [ "$round" -eq 1 ]; then
folder_path=${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/qwen3_8b_512_tasks_20steps_temperature0.7/round_$round
bootstrap_round_1_data "$folder_path"
else
folder_path=${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}/round_$round
fi

conda activate verl-agent
cd "$WEBVOYAGER_ROOT"
python construct_dataset.py --algo step_grpo_correct_traj --correct_traj_only --round $round --main_folder $folder_path --max_tasks ${MAX_TASKS}
conda deactivate

conda activate verl
cd "$VERL_ROOT"
python "$VERL_ROOT/examples/data_preprocess/webvoyager_step_to_parquet.py" --input_json ${folder_path}/step_grpo_correct_traj_round_${round}.json --output_dir ${folder_path}/verl
# fi

########################################################
### start the PPO training
########################################################
pkill python || true
ray stop || true

conda activate verl
EXPERIMENT_NAME=${FULL_NAME}_round${round}
PROJECT_NAME=cua_rl

# Must match the DataProto batch size (see DataProto.batch size).
TRAIN_BATCH_SIZE=64
N_GPU=8

# Must match how you generated the DataProto.
PROMPT_LENGTH=8192
RESPONSE_LENGTH=512

# Rollout parameters only used for naming and consistency checks.
ROLLOUT_N=8

if [ "$round" -eq 1 ]; then
  MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
else
ckpt_path=${PRO_CUA_CKPT_ROOT}/qwen3-vl-8b/${FULL_NAME}/round_$((round-1))
tracker_file="${ckpt_path}/latest_checkpointed_iteration.txt"

if [[ -f "$tracker_file" ]]; then
LAST_STEP="$(tr -d '[:space:]' < "$tracker_file")"
else
# fallback: pick max global_step_* directory
LAST_STEP="$(find "$ckpt_path" -maxdepth 1 -type d -name 'global_step_*' \
    | sed -E 's/.*global_step_([0-9]+)/\1/' \
    | sort -n | tail -1)"
fi

echo "Using checkpoint step: $LAST_STEP"
MODEL_PATH=${PRO_CUA_CKPT_ROOT}/qwen3-vl-8b/${FULL_NAME}/round_$((round-1))/global_step_${LAST_STEP}/actor_hf
fi

SECONDS=0
# CURRENT_LR=$(python3 -c "
# import math
# lr_max = float('${BASE_LR}')
# lr_min = float('${MIN_LR}')
# round_idx = int('${round}') - 1
# total = int('${TOTAL_ROUNDS}')
# print(f'{lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(round_idx * math.pi / total)):.8f}')
# ")
CURRENT_LR=${BASE_LR}
echo "Current LR: ${CURRENT_LR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${folder_path}/verl/train.parquet \
  data.val_files=${folder_path}/verl/train.parquet \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature=1 \
  data.prompt_key=prompt \
  data.image_key=images \
  data.train_batch_size=64 \
  data.max_prompt_length=8192 \
  data.max_response_length=512 \
  actor_rollout_ref.rollout.max_model_len=12288 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=${CURRENT_LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path=${VERL_ROOT}/examples/data_preprocess/webvoyager_reward.py \
  reward.custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=$EXP_NAME \
  trainer.experiment_name=${EXP_NAME}_${MODEL_NAME}_${MAX_TASKS}tasks_round_${round} \
  trainer.n_gpus_per_node=${N_GPU} \
  trainer.nnodes=1 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.save_freq=50 \
  trainer.test_freq=0 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.default_local_dir=${PRO_CUA_CKPT_ROOT}/qwen3-vl-8b/${FULL_NAME}/round_${round}


echo "PPO time cost for round ${round}: ${SECONDS}s"

########################################################
### merge the checkpoints
########################################################
ckpt_path=${PRO_CUA_CKPT_ROOT}/qwen3-vl-8b/${FULL_NAME}/round_${round}

tracker_file="${ckpt_path}/latest_checkpointed_iteration.txt"

if [[ -f "$tracker_file" ]]; then
  LAST_STEP="$(tr -d '[:space:]' < "$tracker_file")"
else
  # fallback: pick max global_step_* directory
  LAST_STEP="$(find "$ckpt_path" -maxdepth 1 -type d -name 'global_step_*' \
    | sed -E 's/.*global_step_([0-9]+)/\1/' \
    | sort -n | tail -1)"
fi

if [[ -z "${LAST_STEP:-}" ]]; then
  echo "No checkpoint step found under $ckpt_path"
  exit 1
fi

echo "Using checkpoint step: $LAST_STEP"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $ckpt_path/global_step_${LAST_STEP}/actor \
    --target_dir $ckpt_path/global_step_${LAST_STEP}/actor_hf

# rm -rf $ckpt_path/global_step_${LAST_STEP}/actor

conda deactivate

########################################################
### webvoyager evaluation
########################################################
conda activate verl-agent

cd "$WEBVOYAGER_ROOT"
export PATH="$CONDA_PREFIX/bin:$PATH" 

pkill -f chrome || true
rm -rf /tmp/.org.chromium.Chromium.*
model_path=${ckpt_path}/global_step_${LAST_STEP}/actor_hf

bash scripts/launch_vllm_servers.sh $model_path

sleep 120
nohup python flask_router.py > logs/flask_router.log 2>&1 &

pkill -f chrome || true
rm -rf /tmp/.org.chromium.Chromium.*

start=`date +%s`
python -u "$WEBVOYAGER_ROOT/run_qwen3_og_parallel_playwright_data_collection_train.py" \
    --test_file "$WEBVOYAGER_ROOT/data/WebVoyager_data_clean.jsonl" \
    --output_dir ${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_data_clean/${FULL_NAME} \
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
### rollout for ${MAX_TASKS} trajectories
########################################################
if [ "$round" -ne "$TOTAL_ROUNDS" ]; then
    start=`date +%s`
    python -u "$WEBVOYAGER_ROOT/run_qwen3_og_parallel_playwright_data_collection_train.py" \
        --test_file "$WEBVOYAGER_ROOT/data/webvoyager_train_data_unique_ids.jsonl" \
        --output_dir ${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME} \
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
### qwen judge
########################################################
bash scripts/launch_vllm_servers.sh
sleep 120
nohup python flask_router.py > logs/flask_router.log 2>&1 &

########################################################
### label the trajectories
########################################################
cd evaluation
DIRS=(
    "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_data_clean/${FULL_NAME}/round_$((round+1))"
    "${PRO_CUA_DATA_ROOT}/qwen3_WebVoyager_train_data_unique_ids_train/${FULL_NAME}/round_$((round+1))"
    # Add more directories as needed
)

# Loop through each directory
for DIR in "${DIRS[@]}"; do
    echo "Processing directory: $DIR"
    python -u "$WEBVOYAGER_ROOT/evaluation/auto_eval_qwen.py" \
        --api_key "${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}" \
        --process_dir "$DIR" \
        --max_attached_imgs 5 > "$DIR/eval.txt"
    echo "Completed: $DIR"
    echo "----------------------------------------"
done
echo "All evaluations completed!"
cd ..

pkill python || true

done
