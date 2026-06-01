#!/bin/bash

# Define array of directories
DIRS=(
    "/mnt/cache/users/t-yifeihe/cua/results/WebVoyager_data_clean/UI-TARS-1.5-7B/base_100_1/20250806_17_58_42"
    # Add more directories as needed
)

# Loop through each directory
for DIR in "${DIRS[@]}"; do
    echo "Processing directory: $DIR"
    python -u auto_eval_parallel.py \
        --api_key YOUR_OPENAI_API_KEY \
        --process_dir "$DIR" \
        --max_attached_imgs 50 > "$DIR/eval.txt"
    echo "Completed: $DIR"
    echo "----------------------------------------"
done

echo "All evaluations completed!"