#!/bin/bash
# DIR=/mnt/italynorth/prm-cache/users/t-yifeihe/cua/results/webvoyager_notime/uitars_50steps_15imgs/20250608_19_05_05
# python -u auto_eval.py \
#     --api_key YOUR_OPENAI_API_KEY \
#     --process_dir $DIR\
#     --max_attached_imgs 50 > $DIR/eval_50.txt

DIR=/mnt/cache/users/t-yifeihe/cua/data/openwebvoyager_full_clean_gpt4o_cua_4rollout/20250804_16_40_06
python -u /data/data/users/t-yifeihe/cua/WebVoyager/evaluation/auto_eval_oai_openwebvoyager.py \
    --api_key YOUR_OPENAI_API_KEY \
    --process_dir $DIR\
    --api_model gpt-4o-global \
    --max_attached_imgs 50 > $DIR/eval.txt

# python -u auto_eval.py \
#     --api_key YOUR_OPENAI_API_KEY \
#     --process_dir ../results/webvoyager/uitars_50steps/20250604_23_18_41 \
#     --max_attached_imgs 50 > eval_results/webvoyager_uitars_50steps.txt
    # --process_dir ../results/examples \