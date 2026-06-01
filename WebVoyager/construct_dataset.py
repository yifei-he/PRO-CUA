import os
import glob
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from prompts import QWEN3_SYSTEM_PROMPT
import base64
import re
import random

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def format_to_conv(process_dir):
    interact_path = os.path.join(process_dir, 'interact_messages.json')
    if not os.path.exists(interact_path):
        print(f'File not found: {interact_path}')
        return 0
    
    with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
        it_messages = json.load(fr)
    
    task_info = it_messages[1]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]

    task_info_msg = task_info.replace('\nObservation:', '<image>\nObservation:')
    
    start_msg = {
        'role': it_messages[1]['role'],
        'content': task_info_msg
    }
    processed_convo = [start_msg]
    image_list = []

    is_last = False

    action_idx = 0
    ss_idx = 1

    msg_idx = -1
    for message in it_messages:
        msg_idx += 1
        if message['role'] == 'assistant':
            if msg_idx + 1 < len(it_messages) and it_messages[msg_idx + 1]['role'] == 'user':
                if isinstance(it_messages[msg_idx + 1]['content'], str) and 'Format ERROR' in it_messages[msg_idx + 1]['content']:
                    # Drop this step entirely to keep conversations/images aligned.
                    continue

            # use preloaded annotated screenshot
            image_list.append(f"{process_dir}/screenshot{action_idx+1}.png")

            action_idx += 1

            processed_convo.append({
                "from": "assistant",
                "value": message['content']
            })

            processed_convo.append({
                "from": "user",
                "value": "<image>Please analyze the attached screenshot and give the Thought and Action."
            })

            if is_last:
                break
    
    processed_convo = processed_convo[:-1]
    
    return {
        "system": QWEN3_SYSTEM_PROMPT,
        "conversations": processed_convo,
        "images": image_list,
    }


# --- Configuration ---
def construct_dataset(args):
    # results_path = os.path.join(args.main_folder, "results.json")

    # Better set correct_traj_only to be False because we can filter the correct one later

    # Load evaluation results
    # with open(results_path, 'r') as f:
    #     eval_results = json.load(f)

    # Collect all JSON files across subfolders
    subfolders = [f.path for f in os.scandir(args.main_folder) if f.is_dir()]
    json_files = []
    for sub in subfolders:
        json_file = os.path.join(sub, args.judge_file_name)
        if not os.path.exists(json_file):
            # print(f"Skipping {json_file} as it does not exist.")
            continue
        json_files.extend(glob.glob(os.path.join(sub, args.judge_file_name)))

    json_files = sorted(json_files)
    if args.max_tasks is not None and args.max_tasks < len(json_files):
        rng = random.Random(args.seed)
        json_files = rng.sample(json_files, args.max_tasks)

    # Define per-file processing function
    def process_json(json_file):
        process_dir = '/'.join(json_file.split('/')[:-1])
        # Skip tasks with zero eval score
        task_name = json_file.split('/')[-2]
        # task_name = os.path.splitext(os.path.basename(json_file))[0]
        # if args.correct_traj_only and eval_results[task_name] == 0:
        #     return None

        # Load the conversation JSON
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        # If the loaded file already contains processed conversations/images
        # (e.g., interact_messages_with_prm_score.json), keep it to preserve scores.
        if args.judge_file_name == 'interact_messages.json':
            raw_data = format_to_conv(process_dir)
        
        if isinstance(raw_data, int):
            print(f"Skipping {json_file} as the content is an integer.")
            return None

        # Filter out steps with format errors by building new lists
        original_conversations = raw_data['conversations']
        original_images = raw_data['images']
        
        new_conversations = []
        new_images = []

        # Iterate over conversation pairs (user, assistant)
        # Start from 1 to skip system prompt
        i = 0
        image_idx = 0
        while i < len(original_conversations) - 1:
            user_msg = original_conversations[i]
            assistant_msg = original_conversations[i+1]

            # Check for malformed assistant message
            # if 'taskApple--extend--45-3' in task_name:
            if 'from' in assistant_msg and assistant_msg['from'] == 'assistant' and assistant_msg['value'].endswith('Action: '):
                print(f"Found and removed malformed step in {task_name}")
                # Skip this pair
            else:
                # If valid, add the pair to the new list
                new_conversations.append(user_msg)
                new_conversations.append(assistant_msg)
                # And add the corresponding image
                if image_idx < len(original_images):
                    new_images.append(original_images[image_idx])
            
            i += 2
            image_idx += 1

        raw_data['conversations'] = new_conversations
        raw_data['images'] = new_images
        
        return {
            'system': raw_data['system'],
            'conversations': raw_data['conversations'],
            'images': raw_data['images'],
            'task_name': task_name,
        }

    # --- Parallel Execution---
    max_workers = 64  # adjust based on your environment
    all_datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_json, jf): jf for jf in json_files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_datasets.append(result)

    # Summary and Save
    print(f"{len(all_datasets)} out of {len(json_files)} samples retained")

    # output_path = os.path.join(main_folder, 'github_rollout_sft_correct_traj.json')
    output_path = os.path.join(args.main_folder, f'rollout_sft_{args.algo}_round_{args.round}.json')
    with open(output_path, 'w') as f:
        json.dump(all_datasets, f, indent=4, ensure_ascii=False)


def create_sliding_window_data(example, window_size=5, correct_traj_only=False, correct_step_only=True, finished_traj_only=True, action_only=False):
    conversations = example['conversations']
    if not conversations:
        print(f"Skipping empty conversations for task {example['task_name']}")
        return
    if conversations[0]['value']:
        if '<image>' in conversations[0]['value']:
            conversations[0] = {
                'from': conversations[0]['from'],
                'value': conversations[0]['value']
            }
        else:
            conversations[0] = {
                'from': conversations[0]['from'],
                'value': conversations[0]['value'] + '<image>Please analyze the attached screenshot and give the Thought and Action.'
            }
    elif conversations[0]['content']:
        if '<image>' in conversations[0]['content']:
            conversations[0] = {
                'from': conversations[0]['role'],
                'value': conversations[0]['content']
            }
        else:
            conversations[0] = {
                'from': conversations[0]['role'],
                'value': conversations[0]['content'] +'<image>Please analyze the attached screenshot and give the Thought and Action.'
            }
    images = example['images']
    
    cur_folder = '/'.join(images[0].split('/')[:-2])
    cur_task = images[0].split('/')[-2]
    results_path = os.path.join(cur_folder, "results.json")

    # Load evaluation results
    with open(results_path, 'r') as f:
        traj_results = json.load(f)

    if correct_traj_only and traj_results[example['task_name']] == 0:
        return
    
    if finished_traj_only:
        if os.path.exists(os.path.join(cur_folder, cur_task, 'screenshot20.png')):
            return

    sliding_windows = []
    contain_video = False
    num_steps = len(conversations) // 2  # Each step = user+assistant

    for step in range(num_steps):
        start_idx = max(0, (step + 1 - window_size) * 2)
        end_idx = (step + 1) * 2
        window_conversations = []
        # For mapping user turn index to image index
        cur_user_image_indices = []
        user_turn_count = 0
        for i in range(end_idx):
            conv = conversations[i].copy()
            if conv['from'] == 'user' and '<image>' in conv['value']:
                if i >= start_idx:
                    window_conversations.append(conv)
                    cur_user_image_indices.append(user_turn_count)
                else:
                    conv['value'] = conv['value'].replace('<image>', '<SCREENSHOT_REDACTED>')
                    window_conversations.append(conv)
                user_turn_count += 1
            else:
                if action_only and conv['from'] == 'assistant':
                    action_str = 'Action: ' + conv['value'].split('Action: ')[-1]
                    window_conversations.append({
                        'from': conv['from'],
                        'value': action_str,
                        'score': conv.get('score', None)
                    })
                else:
                    window_conversations.append(conv)

        # Now, select the images corresponding to the user turns in the window
        window_images = []
        for idx in cur_user_image_indices:
            if idx < len(images):
                window_images.append(images[idx])
                
        conv = window_conversations[-1]

        if correct_step_only:
            if conv.get('score') is None:
                continue
            
            if conv['from'] == 'assistant' and conv.get('score', 0) < 1:
                if 'finished(content=' in conv['value']:
                    if traj_results[example['task_name']] == 0:
                        continue
                else:
                    continue
        
        if contain_video:
            continue
            
        # Clean conversations - only keep 'from' and 'value' fields
        clean_conversations = []
        for convo in window_conversations:
            clean_convo = {
                'from': convo.get('from'),
                'value': convo.get('value')
            }
            # Skip entries where both from and value are None/null
            if clean_convo['from'] is not None or clean_convo['value'] is not None:
                clean_conversations.append(clean_convo)
                
        sliding_windows.append({
            'system': example['system'],
            'conversations': clean_conversations,
            'images': window_images
        })

    return sliding_windows

from datasets import load_dataset

# Those parameters are very important, please set them carefully
def main(args):
    window_size = 1

    construct_dataset(args)
    ds = load_dataset("json", data_files=f'{args.main_folder}/rollout_sft_{args.algo}_round_{args.round}.json', split="train")

    sliding_window_path = f"{args.main_folder}/{args.algo}_round_{args.round}.json"
    print(f"File path: {sliding_window_path}")

    print(len(ds))
    all_sliding_windows = []

    for example in ds:
        sliding_windows = create_sliding_window_data(example, window_size=window_size, correct_traj_only=args.correct_traj_only, correct_step_only=args.correct_step_only, finished_traj_only=args.finished_traj_only)
        if sliding_windows is not None:
            all_sliding_windows.extend(sliding_windows)

    print(len(all_sliding_windows))
    print(sliding_window_path)

    with open(sliding_window_path, 'w', encoding='utf-8') as f:
        json.dump(all_sliding_windows, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type=str, default='results')
    parser.add_argument('--judge_file_name', type=str, default='interact_messages.json')
    parser.add_argument('--correct_traj_only', action='store_true')
    parser.add_argument('--correct_step_only', action='store_true')
    parser.add_argument('--finished_traj_only', action='store_true')
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
