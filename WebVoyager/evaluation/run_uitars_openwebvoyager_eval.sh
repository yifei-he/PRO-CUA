#!/bin/bash
# Define array of directories
DIRS=(
    "/mnt/cache/users/t-yifeihe/cua/results/online_mind2web_clean/Qwen2.5-VL-7B-Instruct/openwebvoyager_full_v2_gpt4o_cua_correct_step_correct_traj_sliding_window_1_2epoch_1e-5_64gpu/20250806_20_36_38"
    # Add more directories as needed
)

# Loop through each directory
for DIR in "${DIRS[@]}"; do
    echo "Processing directory: $DIR"
    python -u auto_eval_uitars_openwebvoyager.py \
        --api_key YOUR_OPENAI_API_KEY \
        --process_dir "$DIR" \
        --max_attached_imgs 50 > "$DIR/eval.txt"
    echo "Completed: $DIR"
    echo "----------------------------------------"
done

echo "All evaluations completed!"
