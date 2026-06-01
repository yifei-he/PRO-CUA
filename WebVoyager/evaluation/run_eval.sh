#!/bin/bash
DIR=/mnt/italynorth/prm-cache/users/t-yifeihe/cua/results/webvoyager_notime/gpt-4o_som_50steps_parallel/20250613_17_13_34
python -u auto_eval.py \
    --api_key YOUR_OPENAI_API_KEY \
    --process_dir $DIR\
    --max_attached_imgs 50 > $DIR/eval.txt