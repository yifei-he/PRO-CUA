import argparse
import os
import json
import time
import re
import base64
import concurrent.futures
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFont
import random

from openai import OpenAI
from openai import AzureOpenAI
from gpt_prompts import GPT_STEP_JUDGE
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from map_action import map_processed_oai_action, map_oai_action
from prompts import QWEN3_SYSTEM_PROMPT
from gpt_prompts import GPT_STEP_JUDGE, GPT_THOUGHT_AUGMENTATION, USER_PROMPT, GPT_STEP_JUDGE_REVISED, OPENCUA_PRM_PROMPT, OPENCUA_PRM_PROMPT_NO_HISTORY, OPENCUA_PRM_PROMPT_NO_HISTORY_NO_FINAL_OUTCOME

# from data_visualization import actions_visual, create_zoomed_action_image


def _repeated_assistant_msg_idxs(it_messages, min_repeat=3):
    assistant_msgs = [
        (idx, msg["content"])
        for idx, msg in enumerate(it_messages)
        if msg.get("role") == "assistant"
    ]
    repeated = set()
    if not assistant_msgs:
        return repeated

    streak_content = assistant_msgs[0][1]
    streak_idxs = [assistant_msgs[0][0]]
    for idx, content in assistant_msgs[1:]:
        if content == streak_content:
            streak_idxs.append(idx)
        else:
            if len(streak_idxs) >= min_repeat:
                repeated.update(streak_idxs)
            streak_content = content
            streak_idxs = [idx]
    if len(streak_idxs) >= min_repeat:
        repeated.update(streak_idxs)
    return repeated


def _repeated_scroll_msg_idxs(it_messages, min_repeat=5):
    assistant_msgs = [
        (idx, msg["content"])
        for idx, msg in enumerate(it_messages)
        if msg.get("role") == "assistant"
    ]
    repeated = set()
    if not assistant_msgs:
        return repeated

    streak_idxs = []
    for idx, content in assistant_msgs:
        action = json.loads(content.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        if action.get("arguments", {}).get("action") == "scroll":
            streak_idxs.append(idx)
        else:
            if len(streak_idxs) >= min_repeat:
                repeated.update(streak_idxs)
            streak_idxs = []
    if len(streak_idxs) >= min_repeat:
        repeated.update(streak_idxs)
    return repeated


def _images_are_identical(image_path_a, image_path_b):
    if not (os.path.exists(image_path_a) and os.path.exists(image_path_b)):
        return False

    with Image.open(image_path_a) as img_a, Image.open(image_path_b) as img_b:
        img_a = img_a.convert("RGB")
        img_b = img_b.convert("RGB")
        if img_a.size != img_b.size:
            return False
        return ImageChops.difference(img_a, img_b).getbbox() is None


def _mark_consecutive_identical_image_scores(processed_convo, image_list):
    assistant_entries = [
        entry for entry in processed_convo
        if entry.get("from") == "assistant"
    ]

    for idx in range(1, min(len(assistant_entries), len(image_list))):
        if _images_are_identical(image_list[idx - 1], image_list[idx]):
            for repeated_entry in (assistant_entries[idx - 1], assistant_entries[idx]):
                repeated_entry["score"] = 0
                existing_judge = repeated_entry.get("judge")
                image_judge = "Auto score: consecutive screenshots are visually identical."
                if existing_judge:
                    if image_judge not in existing_judge:
                        repeated_entry["judge"] = f"{existing_judge} {image_judge}"
                else:
                    repeated_entry["judge"] = image_judge


def auto_eval_by_gpt4v(process_dir, openai_client, model_name='Qwen/Qwen3-VL-8B-Instruct', final_outcome=None, include_final_outcome=False):
    # print("Processing ", process_dir)
    if os.path.exists(os.path.join(process_dir, 'concise_interact_messages.json')):
        interact_path = os.path.join(process_dir, 'concise_interact_messages.json')
        with open(interact_path, 'r') as fr:
            it_messages = json.load(fr)
    else:
        interact_path = os.path.join(process_dir, 'interact_messages.json')
        with open(interact_path, 'r') as fr:
            it_messages = json.load(fr)

    # interact_path = os.path.join(process_dir, 'interact_messages.json')
    # if not os.path.exists(interact_path):
    #     print(f'File not found: {interact_path}')
    #     return 0
    
    # with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
    #     it_messages = json.load(fr)
    repeated_msg_idxs = _repeated_assistant_msg_idxs(it_messages, min_repeat=3)
    repeated_scroll_msg_idxs = _repeated_scroll_msg_idxs(it_messages, min_repeat=5)
    
    task_info = it_messages[1]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info
    pattern = r"(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    action_idx = 0
    whole_content_img = []
    sliding_window = []
    previous_actions = []

    task_info_msg = task_info.replace('\nObservation:', '<image>\nObservation:')
    
    start_msg = {
        'role': it_messages[1]['role'],
        'content': task_info_msg
    }
    processed_convo = [start_msg]
    image_list = []

    is_last = False

    msg_idx = -1
    history_actions = []
    for message in it_messages:
        msg_idx += 1
        if message['role'] == 'assistant':
            if msg_idx + 1 < len(it_messages) and it_messages[msg_idx + 1]['role'] == 'user':
                if isinstance(it_messages[msg_idx + 1]['content'], str) and 'Format ERROR' in it_messages[msg_idx + 1]['content']:
                    processed_convo.append({
                        "from": "assistant",
                        "value": message['content'],
                        "score": 0,
                        "judge": 'Format error'
                    })

                    processed_convo.append({
                        "from": "user",
                        "value": "<image>Please analyze the attached screenshot and give the Thought and Action."
                    })
                    action_idx += 1
                    image_list.append(f"{process_dir}/screenshot{action_idx}.png")

                    history_actions.append(json.loads(message['content'].split('<tool_call>\n')[1].split('\n</tool_call>')[0]))
                    continue
                
            if msg_idx in repeated_msg_idxs or msg_idx in repeated_scroll_msg_idxs:
                processed_convo.append({
                    "from": "assistant",
                    "value": message['content'],
                    "score": 0,
                    "judge": 'Auto score: repeated identical assistant action in streak >= 3.' if msg_idx in repeated_msg_idxs else 'Auto score: repeated scroll action in streak >= 6.'
                })

                processed_convo.append({
                    "from": "user",
                    "value": "<image>Please analyze the attached screenshot and give the Thought and Action."
                })
                action_idx += 1
                image_list.append(f"{process_dir}/screenshot{action_idx}.png")

                history_actions.append(json.loads(message['content'].split('<tool_call>\n')[1].split('\n</tool_call>')[0]))
                continue

            image_list.append(f"{process_dir}/screenshot{action_idx+1}.png")

            action_idx += 1

            processed_convo.append({
                "from": "assistant",
                "value": message['content'],
                "score": 1
            })

            processed_convo.append({
                "from": "user",
                "value": "<image>Please analyze the attached screenshot and give the Thought and Action."
            })

            if is_last:
                break

    
    processed_convo = processed_convo[:-1]
    _mark_consecutive_identical_image_scores(processed_convo, image_list)
    
    return {
        "system": QWEN3_SYSTEM_PROMPT,
        "conversations": processed_convo,
        "images": image_list,
    }
## Removed process_single_folder; auto_eval_by_gpt4v returns directly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, default='results')
    parser.add_argument('--max-workers', type=int, default=8, help='threads for folder-level parallelism')
    parser.add_argument('--batch-size', type=int, default=32, help='folders per round to process')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Model to use for evaluation')
    parser.add_argument('--finished-only', action='store_true', help='only process folders that have finished the task')
    parser.add_argument('--correct-traj-only', action='store_true', help='only process folders that have correct trajectories')
    parser.add_argument('--include-final-outcome', action='store_true', help='do not use final outcome in the prompt')
    args = parser.parse_args()

    # output_file_name = 'interact_messages_with_prm_score_thought_with_final_outcome.json' if args.include_final_outcome else 'interact_messages_with_prm_score_thought.json'
    output_file_name = 'interact_messages_dedup.json'

    client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",  # Dummy key to satisfy the client
        )
    

    main_folder = args.process_dir
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    result_file = os.path.join(main_folder, 'results.json')
    # read from result_file
    with open(result_file, 'r') as fr:
        eval_res = json.load(fr)

    # Filter folders as needed
    filtered_folders = []
    for folder in subfolders:
        # if 'GitHub' in folder:
        #     continue
        # output_path = os.path.join(folder, output_file_name)
        if os.path.exists(output_file_name):
            print(f'Skipping {folder} as output already exists')
            continue
        
        interact_path = os.path.join(folder, 'interact_messages.json')
        if not os.path.exists(interact_path):
            print(f'Skipping {folder} as no interaction exists')
            continue
    
        # check whether screenshot100.png exists in the folder
        if os.path.exists(os.path.join(folder, 'screenshot50.png')) and args.finished_only:
            print(f'Skipping {folder} as it does not finish the task')
            continue
    
        if args.correct_traj_only:
            task_name = folder.split('/')[-1]
            if eval_res[task_name] == 0:
                continue
            
        
        filtered_folders.append(folder)
    # filtered_folders = ['/data/common/yifei/cua/data/qwen3_WebVoyager_train_data_unique_ids_train/step_filter/round_2/taskAmazon--519-1']
    # print(eval_res.get(os.path.basename(filtered_folders[0]), 0))
    
    print(f'Found {len(filtered_folders)} folders to process')

    completed_count = 0
    for start_idx in range(0, len(filtered_folders), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(filtered_folders))
        round_folders = filtered_folders[start_idx:end_idx]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_folder = {
                executor.submit(auto_eval_by_gpt4v, folder, client, args.model, eval_res.get(os.path.basename(folder), 0), args.include_final_outcome): folder
                for folder in round_folders
            }
            for future in concurrent.futures.as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    full_output = future.result()
                    if isinstance(full_output, dict):
                        # save output JSON
                        out_path = os.path.join(folder, output_file_name)
                        with open(out_path, 'w') as fw:
                            json.dump(full_output, fw, indent=4, ensure_ascii=False)
                        completed_count += 1
                        print(f'✓ Completed {completed_count}/{len(filtered_folders)}: {folder}')
                    else:
                        print(f'✗ No output for: {folder}')
                except Exception as exc:
                    print(f'✗ Error processing {folder}: {exc}')
    
    print(f'Processing complete. Successfully processed {completed_count}/{len(filtered_folders)} folders.')


if __name__ == '__main__':
    main()
