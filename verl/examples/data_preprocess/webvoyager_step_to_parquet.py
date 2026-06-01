#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Convert WebVoyager step-level JSON data into verl RLHF parquet format.

Input schema (per item, expected):
- system: str
- conversations: list[{"from": "user"|"assistant", "value": str}, ...]
- images: list[str]  (absolute or local image paths)

Output schema (per item):
- data_source: str
- prompt: list[{"role": "...", "content": "..."}]
- images: list[{"image": "..."}]
- reward_model: {"style": "rule", "ground_truth": str}
- extra_info: {"index": int, ...}

Important behavior:
- The final assistant turn is treated as golden target and stored in
  reward_model.ground_truth.
- The final assistant turn is NOT included in prompt (to avoid leakage).
"""

import argparse
import json
import os
from typing import Any
import re

import datasets


def _map_role(raw_role: str) -> str:
    if raw_role == "assistant":
        return "assistant"
    if raw_role == "system":
        return "system"
    return "user"


def _convert_item(item: dict[str, Any], idx: int, data_source: str) -> dict[str, Any]:
    conversations = item.get("conversations", [])
    if not conversations:
        raise ValueError(f"Sample index={idx} has empty conversations.")

    # Find the last assistant turn as golden response.
    last_asst_idx = None
    for j in range(len(conversations) - 1, -1, -1):
        if conversations[j].get("from") == "assistant":
            last_asst_idx = j
            break
    if last_asst_idx is None:
        raise ValueError(f"Sample index={idx} has no assistant turn.")

    golden_response = conversations[last_asst_idx].get("value", "")
    if not isinstance(golden_response, str) or not golden_response.strip():
        raise ValueError(f"Sample index={idx} has empty golden assistant response.")

    # Count assistant turns instead of raw conversation indices.
    gold_turn_index = sum(1 for turn in conversations[: last_asst_idx + 1] if turn.get("from") == "assistant")

    # Training prompt excludes the golden assistant turn, and keeps full prior trajectory.
    prompt_turns = conversations[:last_asst_idx]

    prompt = []
    system_text = item.get("system", "")
    if isinstance(system_text, str) and system_text.strip():
        prompt.append({"role": "system", "content": system_text})

    for turn in prompt_turns:
        raw_role = turn.get("from", "user")
        value = turn.get("value", "")
        if not isinstance(value, str):
            value = str(value)
        prompt.append({"role": _map_role(raw_role), "content": value})

    image_paths = item.get("images", []) or []
    images = [{"image": img} for img in image_paths if isinstance(img, str) and img.strip()]

    task_info = prompt[1]['content']
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info
    pattern = r"(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    history_actions = []
    for conv in conversations:
        if conv["from"] == "assistant":
            history_actions.append(json.loads(conv['value'].split('<tool_call>\n')[1].split('\n</tool_call>')[0]))

    return {
        "data_source": data_source,
        "prompt": prompt,
        "images": images,
        "reward_model": {"style": "rule", "ground_truth": golden_response},
        "extra_info": {
            "index": idx,
            "num_turns": len(conversations),
            "gold_turn_index": gold_turn_index,
            "task_instruction": task_content,
            "history_actions": history_actions,
            "image_path": image_paths[0]
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to source JSON file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save train/val parquet.")
    parser.add_argument("--data_source", default="webvoyager_step", help="Tag used in output field `data_source`.")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation ratio in [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit samples after conversion, -1 for all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = os.path.abspath(os.path.expanduser(args.input_json))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Input JSON must be a list of examples.")

    converted = []
    for i, item in enumerate(raw_data):
        try:
            converted.append(_convert_item(item, i, args.data_source))
        except Exception as e:
            print(f"[skip] index={i} reason={e}")

    if not converted:
        raise RuntimeError("No valid samples after conversion.")

    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    dataset = datasets.Dataset.from_list(converted)
    print(f"converted samples: {len(dataset)}")

    if args.val_ratio <= 0:
        train_dataset = dataset
        val_dataset = dataset.select([])
    else:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        val_dataset = split["test"]

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"train size: {len(train_dataset)} -> {train_path}")
    print(f"val size: {len(val_dataset)} -> {val_path}")
    if len(train_dataset) > 0:
        ex = train_dataset[0]
        print("sample keys:", list(ex.keys()))
        print("sample prompt turns:", len(ex["prompt"]))
        print("sample images:", len(ex.get("images", [])))
        print("sample gt head:", ex["reward_model"]["ground_truth"][:120].replace("\n", "\\n"))


if __name__ == "__main__":
    main()

# python -c "import pyarrow.parquet as pq; t = pq.read_table('/scratch/yifeihe/cua/data/test/verl/train.parquet'); print(t.slice(10, 1).to_pylist()[0])"

# python /home/yifeihe/cua_step_rl/verl/examples/data_preprocess/webvoyager_step_to_parquet.py --input_json /scratch/yifeihe/cua/data/test/test_step.json --output_dir /scratch/yifeihe/cua/data/test/verl