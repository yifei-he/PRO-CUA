import argparse
import os
import json
import time
import re
import base64
import concurrent.futures
import requests
from PIL import Image, ImageDraw, ImageFont
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



def _extract_score_from_content(content):
    score_patterns = [
        r'Expected value:\s*(\d+)',
        r'Expected value:\s*\n\s*(\d+)',
    ]
    for pattern in score_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score

    float_patterns = [
        r'Expected value:\s*(-?\d+(?:\.\d+)?)',
        r'Expected value:\s*\n\s*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in float_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            val = float(match.group(1))
            if -1.0 <= val <= 1.0:
                return max(0, min(10, round((val + 1) * 5)))

    correct_match = re.search(
        r'last_step_correct"\s*:\s*(true|false)',
        content,
        re.IGNORECASE
    )
    redundant_match = re.search(
        r'last_step_redundant"\s*:\s*(true|false)',
        content,
        re.IGNORECASE
    )
    if correct_match:
        is_correct = correct_match.group(1).lower() == 'true'
        is_redundant = redundant_match and redundant_match.group(1).lower() == 'true'
        if is_correct and not is_redundant:
            return 1
        if is_correct and is_redundant:
            return 0.5
        return 0

    return None


def _call_api_with_retry(client, model_name, messages, process_dir, call_type="thought"):
    """Helper function to make API calls with retry and fallback logic."""
    retry = 0
    
    while True:
        retry += 1
        if retry == 5:
            print(f'{process_dir} {call_type} call failed after 20 attempts')
            return None
        
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, seed=42, temperature=0
            )
            
            content = response.choices[0].message.content
            
            # For judge calls, we need to validate the score
            if call_type == "judge":
                score = _extract_score_from_content(content)
                if score is not None:
                    return content, score  # Return both content and score
                # else:
                #     # Score extraction failed, treat as a retryable error
                #     print(f"Warning: No valid score found in judge response, retrying... (attempt {retry}/5)")
                #     time.sleep(5)
                #     continue # continue to next retry iteration
            
            return content # For thought calls

        except Exception as e:
            print(f"Error during {call_type} call: {e} with {model_name} on {process_dir}")
            time.sleep(10)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



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


def _repeated_scroll_msg_idxs(it_messages, min_repeat=6):
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
    repeated_scroll_msg_idxs = _repeated_scroll_msg_idxs(it_messages, min_repeat=6)
    # print(repeated_msg_idxs, repeated_scroll_msg_idxs)
    # x
    
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

    # preload base64-encoded screenshots
    annot_dir = os.path.join(process_dir, 'annotated_screenshots')
    annotated_b64 = {}
    if os.path.isdir(annot_dir):
        for fn in os.listdir(annot_dir):
            m = re.match(r'screenshot(\d+)\.png', fn)
            if m:
                idx = int(m.group(1))
                with open(os.path.join(annot_dir, fn), 'rb') as f:
                    annotated_b64[idx] = base64.b64encode(f.read()).decode('utf-8')
    # zoom_dir = os.path.join(process_dir, 'zoomed_screenshots')
    # zoomed_b64 = {}
    # if os.path.isdir(zoom_dir):
    #     for fn in os.listdir(zoom_dir):
    #         m = re.match(r'screenshot(\d+)\.png', fn)
    #         if m:
    #             idx = int(m.group(1))
    #             with open(os.path.join(zoom_dir, fn), 'rb') as f:
    #                 zoomed_b64[idx] = base64.b64encode(f.read()).decode('utf-8')

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

            # use preloaded annotated screenshot
            b64_img = annotated_b64.get(action_idx)
            cur_img = {'type':'image_url','image_url':{'url':f"data:image/png;base64,{b64_img}"}}

            # Add to sliding window and maintain size of 3
            sliding_window.append(cur_img)
            if len(sliding_window) > 3:
                sliding_window.pop(0)  # Remove oldest image

            # use preloaded zoomed screenshot if available
            # if action_idx in zoomed_b64:
            #     cur_zoomed_img = {'type':'image_url','image_url':{'url':f"data:image/png;base64,{zoomed_b64[action_idx]}"}}
            # else:
            #     cur_zoomed_img = None
            whole_content_img.append(cur_img)

            if include_final_outcome:
                user_prompt_tmp = OPENCUA_PRM_PROMPT_NO_HISTORY.replace('{instruction}', task_content)
            else:
                user_prompt_tmp = OPENCUA_PRM_PROMPT_NO_HISTORY_NO_FINAL_OUTCOME.replace('{instruction}', task_content)

            user_prompt_tmp = user_prompt_tmp.replace('{step_index}', str(action_idx))
            user_prompt_tmp = user_prompt_tmp.replace('{action_code}', json.dumps(message))
            user_prompt_tmp = user_prompt_tmp.replace('{history_actions}', json.dumps(history_actions))

            if final_outcome is not None and include_final_outcome:
                user_prompt_tmp = user_prompt_tmp.replace('{final_outcome}', 'SUCCESS' if final_outcome == 1 else 'FAILURE')
            
            # print(user_prompt_tmp)

            next_img = None
            if os.path.exists(os.path.join(process_dir, f'screenshot{action_idx+2}.png')):
                next_img = {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/png;base64,{encode_image(os.path.join(process_dir, f'screenshot{action_idx+2}.png'))}"
                    }
                }

            history_imgs = sliding_window[:-1]
            before_img = sliding_window[-1]

            content = [{'type': 'text', 'text': user_prompt_tmp}]
            # if history_imgs:
            #     content.append({'type': 'text', 'text': "History screenshots (older to newer):"})
            #     for i, img in enumerate(history_imgs, start=1):
            #         content.append({'type': 'text', 'text': f"History {i}:"})
            #         content.append(img)

            content.append({'type': 'text', 'text': "Screenshot BEFORE last action (if not provided, the last action is the first action):"})
            content.append(before_img)
            content.append({'type': 'text', 'text': "Screenshot AFTER last action (if not provided, the last action is the final action):"})
            if next_img is not None:
                content.append(next_img)
            else:
                content.append({'type': 'text', 'text': "No after screenshot available."})
            content.append({'type': 'text', 'text': "Your judgement:\n"})

            judge_messages = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
            
            judge = None
            score = None

            judge_result = _call_api_with_retry(
                openai_client, model_name, judge_messages, process_dir, "judge"
            )

            # Process judge result
            if judge_result is None:
                print(f"Could not get judge for {process_dir}, action {action_idx}. Aborting folder.")
                return None # Or handle as per requirements
            judge, score = judge_result

            if judge is None or score is None:
                print(f"Failed to get thought/judge/score for {process_dir}. Skipping folder.")
                return None
            
            action_str = f"Step {action_idx}:" + message['content'].split('\nAction:')[1]
            previous_actions.append(action_str)
            image_list.append(f"{process_dir}/screenshot{action_idx+1}.png")

            action_idx += 1

            processed_convo.append({
                "from": "assistant",
                "value": message['content'],
                "score": score,
                "judge": judge
            })

            processed_convo.append({
                "from": "user",
                "value": "<image>Please analyze the attached screenshot and give the Thought and Action."
            })

            # Add to history actions
            history_actions.append(json.loads(message['content'].split('<tool_call>\n')[1].split('\n</tool_call>')[0]))

            print(f"Action {action_idx} in {process_dir} gets score {score}")

            if is_last:
                break

    
    processed_convo = processed_convo[:-1]
    
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

    output_file_name = 'interact_messages_with_prm_score_thought_with_final_outcome.json' if args.include_final_outcome else 'interact_messages_with_prm_score.json'

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
        if os.path.exists(os.path.join(folder, 'screenshot20.png')) and args.finished_only:
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
    
    # # Process folders in parallel
    # completed_count = 0
    # # folder-level parallelism using configurable worker count
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     # Submit all tasks by calling auto_eval_by_gpt4v directly
    #     future_to_folder = {
    #         executor.submit(auto_eval_by_gpt4v, folder, client, args.model): folder
    #         for folder in filtered_folders
    #     }
        
    #     # Process completed tasks as they finish
    #     for future in concurrent.futures.as_completed(future_to_folder):
    #         folder = future_to_folder[future]
    #         # try:
    #         full_output = future.result()
    #         if isinstance(full_output, dict):
    #             # save output JSON
    #             out_path = os.path.join(folder, 'interact_messages_with_prm_score.json')
    #             with open(out_path, 'w') as fw:
    #                 json.dump(full_output, fw, indent=4, ensure_ascii=False)
    #             completed_count += 1
    #             print(f'✓ Completed {completed_count}/{len(filtered_folders)}: {folder}')
    #         else:
    #             print(f'✗ No output for: {folder}')
    #         # except Exception as exc:
    #         #     print(f'✗ Error processing {folder}: {exc}')
    
    print(f'Processing complete. Successfully processed {completed_count}/{len(filtered_folders)} folders.')


if __name__ == '__main__':
    main()
