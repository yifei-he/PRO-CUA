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
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, InteractiveBrowserCredential, AzureDeveloperCliCredential, CertificateCredential
from gpt_prompts import GPT_STEP_JUDGE
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from map_action import map_processed_oai_action, map_oai_action
from prompts import COMPUTER_USE_DOUBAO
from gpt_prompts import GPT_STEP_JUDGE, GPT_THOUGHT_AUGMENTATION, USER_PROMPT, GPT_STEP_JUDGE_REVISED, GPT_STEP_JUDGE_CONCISE

from data_visualization import actions_visual, create_zoomed_action_image


class LLMClient:
    _ENDPOINT = 'https://fe-26.qas.bing.net/'
    _SCOPES = ['https://substrate.office.com/llmapi/LLMAPI.dev']
    _API_RESPONSES = 'chat/completions'
    _client_id = '58f2b058-369b-42bc-8060-183aac158dd5'
    _tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
    _certificate_path = '/mnt/cache/users/chawlapranit/pika/syntheticturingdata-SyntheticDataContext-SNI-20250709.pfx'
    _scenario_id = 'e42391ad-b1d3-441d-9f35-1ce23b26a25f'

    def __init__(self, endpoint=None):
        # reuse HTTP session and credential for pooling
        self.endpoint = endpoint or LLMClient._ENDPOINT
        self.session = requests.Session()
        self.credential = CertificateCredential(
            tenant_id=self._tenant_id,
            client_id=self._client_id,
            certificate_path=self._certificate_path,
            send_certificate_chain=True
        )

    def send_request(self, model_name, request, api_version=None):
        token = self._get_token()
        headers = {
            'Content-Type':'application/json',
            'Authorization': 'Bearer ' + token,
            'X-ModelType': model_name
        }
        headers["X-ScenarioGUID"] = self._scenario_id
        body = json.dumps(request).encode('utf-8')
        # build URL and reuse session
        url = self.endpoint.rstrip('/') + '/' + LLMClient._API_RESPONSES
        if api_version:
            url += f"?api-version={api_version}"
        resp = self.session.post(url, data=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _get_token(self):
        return self.credential.get_token("https://substrate.office.com/llmapi/.default").token

    def chat_completions_create(self, model, messages, seed=None, temperature=None):
        """Wrapper to make LLMClient compatible with OpenAI API interface"""
        request_data = {
            "messages": messages,
        }
        
        response = self.send_request(model, request_data)
        
        # Convert response to OpenAI-like format
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response['choices'][0]['message']['content'])


def _call_api_with_retry(client, fallback_client, model_name, api_models, messages, process_dir, call_type="thought"):
    """Helper function to make API calls with retry and fallback logic."""
    retry = 0
    client_type = 'azure'
    model_idx = random.randint(0, len(api_models) - 1)
    
    while True:
        retry += 1
        if retry == 20:
            print(f'{process_dir} {call_type} call failed after 20 attempts')
            return None
        
        try:
            cur_model = api_models[model_idx % len(api_models)]
            
            if model_name == 'gpt-4o':
                response = client.chat.completions.create(
                    model=cur_model, messages=messages, seed=42, temperature=0
                )
            elif model_name == 'o4-mini':
                if client_type == 'fallback':
                    response = fallback_client.chat_completions_create(
                        model='dev-o4-mini-2025-04-16', messages=messages, seed=42
                    )
                else:
                    response = client.chat.completions.create(
                        model=cur_model, messages=messages, seed=42
                    )
            
            content = response.choices[0].message.content
            
            # For judge calls, we need to validate the score
            if call_type == "judge":
                score = None
                score_patterns = [
                    r'Expected value:\s*(\d+)',
                    r'Expected value:\s*\n\s*(\d+)',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        score = int(match.group(1))
                        if 0 <= score <= 10:
                            break
                
                if score is not None:
                    return content, score  # Return both content and score
                else:
                    # Score extraction failed, treat as a retryable error
                    print(f"Warning: No valid score found in judge response, retrying... (attempt {retry}/20)")
                    # Fallback logic for score retry
                    if model_name == 'o4-mini':
                        client_type = 'azure' if client_type == 'fallback' else 'fallback'
                        print(f"Switching to {client_type} client for score retry...")
                    model_idx += 1
                    time.sleep(5)
                    continue # continue to next retry iteration
            
            return content # For thought calls

        except Exception as e:
            print(f"Error during {call_type} call: {e} with {cur_model} on {process_dir}")
            if "ResponsibleAIPolicyViolation" in str(e) or "content_filter" in str(e):
                print(f"Content filter triggered during {call_type} call. Stopping.")
                return "CONTENT_FILTER_TRIGGERED"

            print(f"{call_type.capitalize()} API call failed ({client_type}): {e}")
            if model_name == 'o4-mini':
                client_type = 'azure' if client_type == 'fallback' else 'fallback'
                print(f"Switching to {client_type} client...")
            model_idx += 1
            time.sleep(10)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def auto_eval_by_gpt4v(process_dir, openai_client, model_name='o4-mini'):
    interact_path = os.path.join(process_dir, 'interact_messages.json')
    if not os.path.exists(interact_path):
        print(f'File not found: {interact_path}')
        return 0
    
    with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
        it_messages = json.load(fr)
    
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
    zoom_dir = os.path.join(process_dir, 'zoomed_screenshots')
    annotated_b64 = {}
    zoomed_b64 = {}
    if os.path.isdir(annot_dir):
        for fn in os.listdir(annot_dir):
            m = re.match(r'screenshot(\d+)\.png', fn)
            if m:
                idx = int(m.group(1))
                with open(os.path.join(annot_dir, fn), 'rb') as f:
                    annotated_b64[idx] = base64.b64encode(f.read()).decode('utf-8')
    if os.path.isdir(zoom_dir):
        for fn in os.listdir(zoom_dir):
            m = re.match(r'screenshot(\d+)\.png', fn)
            if m:
                idx = int(m.group(1))
                with open(os.path.join(zoom_dir, fn), 'rb') as f:
                    zoomed_b64[idx] = base64.b64encode(f.read()).decode('utf-8')
    action_idx = 0

    msg_idx = -1
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
                    continue

            # use preloaded annotated screenshot
            b64_img = annotated_b64.get(action_idx)
            cur_img = {'type':'image_url','image_url':{'url':f"data:image/png;base64,{b64_img}"}}

            # Add to sliding window and maintain size of 3
            sliding_window.append(cur_img)
            if len(sliding_window) > 3:
                sliding_window.pop(0)  # Remove oldest image

            # use preloaded zoomed screenshot if available
            if action_idx in zoomed_b64:
                cur_zoomed_img = {'type':'image_url','image_url':{'url':f"data:image/png;base64,{zoomed_b64[action_idx]}"}}
            else:
                cur_zoomed_img = None
            whole_content_img.append(cur_img)

            user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
            user_prompt_tmp = user_prompt_tmp.replace('<cur_action>', json.dumps(message))
            user_prompt_tmp = user_prompt_tmp.replace('<previous_actions>', '\n'.join(previous_actions))

            # messages = [
            #     {'role': 'system', 'content': GPT_THOUGHT_AUGMENTATION},
            #     {
            #         'role': 'user',
            #         'content': [
            #             {'type': 'text', 'text': user_prompt_tmp}
            #         ]
            #         + whole_content_img
            #         + [{'type': 'text', 'text': "Your thought process:\n"}]
            #     }
            # ]

            # judge_messages = [
            #     {'role': 'system', 'content': GPT_STEP_JUDGE},
            #     {
            #         'role': 'user',
            #         'content': [
            #             {'type': 'text', 'text': user_prompt_tmp}
            #         ]
            #         + whole_content_img
            #         + [{'type': 'text', 'text': "Your judgement:\n"}]
            #     }
            # ]

            messages = [
                {'role': 'system', 'content': GPT_THOUGHT_AUGMENTATION},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_prompt_tmp}
                    ]
                    + sliding_window  # Use sliding window instead of just current image
                    + [{'type': 'text', 'text': "Your thought process:\n"}]
                }
            ]

            judge_messages = [
                {'role': 'system', 'content': GPT_STEP_JUDGE_REVISED},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_prompt_tmp}
                    ]
                    + sliding_window + ([cur_zoomed_img] if cur_zoomed_img is not None else [])
                    + [{'type': 'text', 'text': "Your judgement:\n"}]
                }
            ]

            if model_name == 'gpt-4o':
                api_models = ['gpt-4o-pika', 'gpt-4o-global', 'gpt-4o']
            elif model_name == 'o4-mini':
                api_models = ['o4-mini']

            # Initialize fallback client for o4-mini (on-demand only)
            fallback_client = LLMClient(None) if model_name == 'o4-mini' else None
            
            thought = None
            judge = None
            score = None

            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     # Submit thought and judge calls to run in parallel
            #     future_thought = executor.submit(
            #         _call_api_with_retry,
            #         openai_client, fallback_client, model_name, api_models, messages, process_dir, "thought"
            #     )
            #     future_judge = executor.submit(
            #         _call_api_with_retry,
            #         openai_client, fallback_client, model_name, api_models, judge_messages, process_dir, "judge"
            #     )

            #     # Wait for results
            #     thought_result = future_thought.result()
            #     judge_result = future_judge.result()
            
            judge_result = _call_api_with_retry(
                openai_client, fallback_client, model_name, api_models, judge_messages, process_dir, "judge"
            )

            # Process judge result
            if judge_result is None:
                print(f"Could not get judge for {process_dir}, action {action_idx}. Aborting folder.")
                return None # Or handle as per requirements
            if judge_result == "CONTENT_FILTER_TRIGGERED":
                 return {
                    "system": COMPUTER_USE_DOUBAO,
                    "conversations": processed_convo[:-1],
                    "images": image_list,
                }
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

            if is_last:
                break
    
    processed_convo = processed_convo[:-1]
    
    return {
        "system": COMPUTER_USE_DOUBAO,
        "conversations": processed_convo,
        "images": image_list,
    }
## Removed process_single_folder; auto_eval_by_gpt4v returns directly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, default='results')
    parser.add_argument('--max-workers', type=int, default=16, help='threads for folder-level parallelism')
    parser.add_argument('--model', type=str, default='o4-mini', choices=['gpt-4o', 'o4-mini'], help='Model to use for evaluation')
    args = parser.parse_args()

    endpoint = "https://turingaoaieastus2.openai.azure.com/"
    # Initialize Azure OpenAI client with Entra ID authentication
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-04-01-preview",
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
        output_path = os.path.join(folder, 'output_thought_judge_4o.json')
        if os.path.exists(output_path):
            print(f'Skipping {folder} as output already exists')
            continue
            
        interact_path = os.path.join(folder, 'interact_messages.json')
        if not os.path.exists(interact_path):
            print(f'Skipping {folder} as no interaction exists')
            continue
        
        # skip folders with imdb, yelp, ikea
        if 'imdb' in folder or 'yelp'in folder or 'discogs' in folder or 'eventbrite' in folder or 'nyc' in folder or 'resy' in folder or 'ryanair' in folder or 'target' in folder:
            print(f'Skipping {folder} as it is an invalid task')
            continue
    
        # # check whether screenshot100.png exists in the folder
        # if os.path.exists(os.path.join(folder, 'screenshot100.png')):
        #     print(f'Skipping {folder} as it does not finish the task')
        #     continue
            
        # task_name = folder.split('/')[-1]
        # if eval_res[task_name] == 0:
        #     continue
        
        filtered_folders.append(folder)
    # filtered_folders = ['/mnt/cache/users/t-yifeihe/cua/data/openwebvoyager_webvoyager_UI-TARS-1.5-7B_8rollout/20250708_21_17_46/taskAllrecipes--20-2']
    
    print(f'Found {len(filtered_folders)} folders to process')
    
    # Process folders in parallel
    completed_count = 0
    # folder-level parallelism using configurable worker count
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks by calling auto_eval_by_gpt4v directly
        future_to_folder = {
            executor.submit(auto_eval_by_gpt4v, folder, client, args.model): folder
            for folder in filtered_folders
        }
        
        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                full_output = future.result()
                if isinstance(full_output, dict):
                    # save output JSON
                    out_path = os.path.join(folder, 'output_thought_judge_4o.json')
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