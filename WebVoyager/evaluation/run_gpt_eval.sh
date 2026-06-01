#!/bin/bash
DIRS=(
    "/scratch/yifeihe/cua/data/qwen3_WebVoyager_train_data_unique_ids_train/qwen3_4b_512_tasks/round_1"
    # Add more directories as needed
)

# Loop through each directory
for DIR in "${DIRS[@]}"; do
    echo "Processing directory: $DIR"
    python -u auto_eval_gpt.py \
        --process_dir "$DIR" \
        --max_attached_imgs 5 > "$DIR/eval.txt"
    echo "Completed: $DIR"
    echo "----------------------------------------"
done
