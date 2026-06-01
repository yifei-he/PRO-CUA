import platform
import argparse
import random
import time
import json
import re
import os
import shutil
import logging

from playwright.async_api import async_playwright

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, COMPUTER_USE_DOUBAO, QWEN3_SYSTEM_PROMPT
from openai import OpenAI
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

import os
import base64
from openai import AzureOpenAI

from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import ast
import asyncio
import concurrent.futures
from cua_utils import CUA_KEY_TO_PLAYWRIGHT_KEY
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from datasets import load_dataset
from urllib.parse import urlparse


def normalize_webgym_task(task):
    website = task.get("website") or task.get("web") or task.get("url")
    question = task.get("task_name") or task.get("ques") or task.get("question")
    task_id = str(task.get("task_id", "")).strip()
    benchmark_name = str(task.get("benchmark_name", "")).strip()

    if not website or not question or not task_id:
        raise ValueError(
            f"WebGym task is missing required fields: website={website}, "
            f"question={question}, task_id={task_id}"
        )

    parsed = urlparse(website)
    website_host = parsed.netloc or parsed.path or benchmark_name or "webgym"
    web_name = benchmark_name or website_host
    normalized_id = f"{web_name}--{task_id}"

    normalized_task = dict(task)
    normalized_task.update({
        "web_name": web_name,
        "id": normalized_id,
        "ques": question,
        "web": website,
        "original_id": task_id,
    })
    return normalized_task


def parse_model_action(model_res):
    if '<tool_call>' in model_res and '</tool_call>' in model_res:
        tool_call = model_res.split('<tool_call>', 1)[1].split('</tool_call>', 1)[0].strip()
        return json.loads(tool_call)

    action_match = re.search(r'Action:\s*(.+)', model_res, re.IGNORECASE | re.DOTALL)
    if not action_match:
        raise ValueError("No Action line found in model response.")

    action_text = action_match.group(1).strip()
    action_line = action_text.splitlines()[0].strip()

    type_match = re.match(r'type\((.*)\)\s*$', action_line, re.IGNORECASE)
    if type_match:
        text = ast.literal_eval(type_match.group(1).strip())
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}

    answer_match = re.match(r'(answer|finished)\((.*)\)\s*$', action_line, re.IGNORECASE)
    if answer_match:
        return {"name": "computer_use", "arguments": {"action": "answer"}}

    simple_action = action_line.lower().strip()
    if simple_action in {"wait", "goback"}:
        return {"name": "computer_use", "arguments": {"action": simple_action}}

    raise ValueError(f"Unsupported action format: {action_line}")

def setup_main_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure directory exists
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_task_logger(task_dir, task_id, trial_id):
    logger = logging.getLogger(f"task_{task_id}_{trial_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(task_dir, 'agent.log'))
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Avoid adding multiple handlers if logger already has one
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ── Playwright Browser Launch ─────────────────────────────────────────────────
async def launch_browser(args):
    pw = await async_playwright().start()
    launch_args = [
        f"--window-size={args.window_width},{args.window_height}",
        "--disable-extensions",
        "--disable-file-system",
    ]
    browser = await pw.chromium.launch(
        chromium_sandbox=True,
        headless=args.headless, 
        args=launch_args,
        env={"DISPLAY": ":0"},
    )
    context = await browser.new_context(
        viewport={"width": args.window_width, "height": args.window_height},
        device_scale_factor=1 if args.force_device_scale else 1
    )
    page = await context.new_page()
    return pw, browser, page


def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        init_msg += f"Please proceed with your Thought and Action."
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. "},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. "},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg

# use vllm openai client
def call_uitars(args, messages, model, processor):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return inputs['input_ids'].shape[1], len(generated_ids_trimmed[0]), False, output_text[0]


def call_gpt4v_api(args, client, messages, model_name):
    retry_times = 0
    
    while True:
        try:
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": args.temperature,
            }
            if args.save_log_prob:
                # vLLM expects an int for logprobs (top-k). Use 1 to get token_ids/logprobs.
                request_kwargs["logprobs"] = max(1, args.completion_logprobs_k)
                if args.prompt_logprobs > 0:
                    request_kwargs["prompt_logprobs"] = args.prompt_logprobs

            openai_response = client.chat.completions.create(
                **request_kwargs,
                # max_completion_tokens=1000,
                # stop=None,
                # stream=False,
                # seed=args.seed,
            ).to_dict()
            
            prompt_tokens = openai_response['usage']['prompt_tokens']
            completion_tokens = openai_response['usage']['completion_tokens']

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)

            elif type(e).__name__ == 'InvalidRequestError':
                gpt_call_error = True
                return None, None, gpt_call_error, None

            else:
                gpt_call_error = True
                return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return None, None, True, None


# ── Action Executors (Playwright) ─────────────────────────────────────────────
async def exec_action_click(info, page):
    x, y = info['x'], info['y']
    
    # Set target=_self if the element has a target attribute
    await page.evaluate("""
        ([x, y]) => {
            const elem = document.elementFromPoint(x, y);
            if (elem && 'target' in elem) {
                elem.setAttribute('target', '_self');
            }
        }
    """, [x, y])

    await page.mouse.click(x, y)
    await asyncio.sleep(3)


async def exec_action_type(info, page):
    await page.keyboard.type(info['content'])
    await page.keyboard.press('Enter')
    await asyncio.sleep(3)


# async def perform_hotkey(page, key_str):
#     keys = key_str.split()
#     mapped_keys = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(k.lower(), k) for k in keys]
#     await page.focus('body')
#     # if 
#     for key in mapped_keys:
#         await page.keyboard.down(key)
#         await asyncio.sleep(0.1)
#     for key in reversed(mapped_keys):
#         await page.keyboard.up(key)
#         await asyncio.sleep(0.1)
#     # for k in keys[:-1]: 
#     #     await page.keyboard.down(k.upper())
#     # await page.keyboard.press(keys[-1].upper())
#     # for k in reversed(keys[:-1]): 
#     #     await page.keyboard.up(k.upper())

async def perform_hotkey(page, key_str: str):
    """
    Perform a hotkey (e.g., 'ctrl a', 'pagedown') on the page.
    Ensures the page has a focused element before sending the keys.

    Args:
        page: Playwright Page object
        key_str: string like 'ctrl a', 'pagedown'
        fallback_focus_selector: selector to focus if no input is focused (defaults to 'body')
    """

    # keys = key_str.strip().lower().split()
    keys = key_str

    mapped_keys = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(key, key) for key in keys]
    for key in mapped_keys:
        await page.keyboard.down(key)
        await asyncio.sleep(1)
    for key in reversed(mapped_keys):
        await page.keyboard.up(key)
        await asyncio.sleep(1)


async def exec_action_scroll(pixels, page):
    # x = box[0]*1000
    # y = box[1]*1000
    # await page.mouse.move(500, 500)
    # dist = args.window_height * 1 // 3
    # scroll_y = dist if info.get("direction") == "down" else -dist
    # scroll_y = info
    await page.evaluate(f"window.scrollBy(0, {-pixels})")
    # delta = dist if info.get('direction') == 'down' else -dist
    # await page.mouse.wheel(0, delta)
    await asyncio.sleep(3)

async def exec_action_hscroll(pixels, page):
    await page.evaluate(f"window.scrollBy({-pixels}, 0)")
    await asyncio.sleep(3)


async def exec_action_drag(info, page):
    # Raw coordinate drag: mouse down, move, mouse up
    x1, y1, x2, y2 = info['x1'], info['y1'], info['x2'], info['y2']
    await page.mouse.move(x1, y1)
    await page.mouse.down()
    await page.mouse.move(x2, y2)
    await page.mouse.up()
    await asyncio.sleep(3)


async def safe_close(browser, pw, task_logger=None):
    if browser is not None:
        try:
            if task_logger:
                task_logger.info("Closing browser...")
            await asyncio.wait_for(browser.close(), timeout=10)
            if task_logger:
                task_logger.info("Browser closed.")
        except Exception as e:
            if task_logger:
                task_logger.warning(f"Browser close failed or timed out: {e}")
    if pw is not None:
        try:
            if task_logger:
                task_logger.info("Stopping Playwright...")
            await asyncio.wait_for(pw.stop(), timeout=10)
            if task_logger:
                task_logger.info("Playwright stopped.")
        except Exception as e:
            if task_logger:
                task_logger.warning(f"Playwright stop failed or timed out: {e}")


async def run_task(task_id, task, trial_id, args, result_dir, client):
    task_dir = os.path.join(result_dir, f'task{task["id"]}-{trial_id}')
    os.makedirs(task_dir, exist_ok=True)
    # setup_logger(task_dir)
    task_logger = get_task_logger(task_dir, task["id"], trial_id)
    task_logger.info(f'########## TASK{task["id"]} Trial {trial_id} ##########')

    pw = None
    browser = None
    page = None
    try:
        pw, browser, page = await launch_browser(args)
        try:
            await page.goto(task['web'], timeout=180000)
            await asyncio.sleep(5)
        except Exception as e:
            task_logger.error(f"Page.goto failed for {task['web']}: {e}")
            return

        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []
        fail_obs = ""
        pdf_obs = ""
        warn_obs = ""
        pattern = r'Thought:|Action:|Observation:'

        messages = [{'role': 'system', 'content': QWEN3_SYSTEM_PROMPT}]
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if args.text_only:
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEXT_ONLY}]
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg = init_msg + obs_prompt

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0
        log_probs = []

        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", use_fast=True)
        patch_size = processor.image_processor.patch_size
        merge_size = processor.image_processor.merge_size
        resized_height, resized_width = smart_resize(
            args.window_height,
            args.window_width,
            factor=patch_size * merge_size,
            min_pixels=patch_size * patch_size * merge_size * merge_size * 16,
            max_pixels=patch_size * patch_size * merge_size * merge_size * 6400,
        )

        while it < args.max_iter:
            print(f'Iter: {it}')
            task_logger.info(f'Iter: {it}')
            it += 1

            img_path = os.path.join(task_dir, f'screenshot{it}.png')
            try:
                await page.screenshot(path=img_path, timeout=60000)
            except Exception as e:
                logging.error(f"Error when taking screenshot: {e}")
                fail_obs = "Internal error: Failed to capture screenshot."
                continue
            b64_img = encode_image(img_path)

            if not fail_obs:
                # message formatting unchanged
                if not args.text_only:
                    msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, None)
                    messages = clip_message_and_obs(messages + [msg], max_img_num=args.max_attached_imgs)
                else:
                    ac_tree, _ = get_webarena_accessibility_tree(page, task_dir)
                    msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree)
                    messages = clip_message_and_obs_text_only(messages + [msg], max_img_num=args.max_attached_imgs)
            else:
                curr_msg = {
                    'role': 'user',
                    'content': fail_obs
                }
                messages.append(curr_msg)

            if not args.text_only:
                messages = clip_message_and_obs(messages, args.max_attached_imgs)
            else:
                messages = clip_message_and_obs_text_only(messages, args.max_attached_imgs)

            if args.model == 'gpt': 
                task_logger.info('Calling gpt4o API...')
                model_name = 'gpt-4o'
            else: 
                task_logger.info('Calling uitars API...')
                model_name = args.model_name

            prompt_tokens, completion_tokens, gpt_call_error, openai_response = call_gpt4v_api(args, client, messages, model_name)
            if openai_response is None:
                print(f"API ERROR: The API call failed in iteration {it}, please try again.")
                continue

            try:
                model_res = openai_response['choices'][0]['message']['content']
            except Exception as e:
                print(openai_response)
                continue
            
            try:
                action = parse_model_action(model_res)
                print(action)
                if 'coordinate' in action['arguments']:
                    coordinate_relative = action['arguments']['coordinate']
                    coordinate_absolute = [coordinate_relative[0] / 1000 * resized_width, coordinate_relative[1] / 1000 * resized_height]
                action_key = action['arguments']['action']
            except Exception as e:
                print(model_res)
                logging.error(f"Error when parsing model response: {e}")
                messages.append({'role': 'assistant', 'content': model_res})
                fail_obs = (
                    "Format ERROR: The Action format is not correct, please follow the format: "
                    "Action: <action_type>(<action_inputs>)\n\n"
                    f"Original model output:\n{model_res}"
                )
                continue
        
            accumulate_prompt_token += prompt_tokens
            accumulate_completion_token += completion_tokens
            task_logger.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
            task_logger.info('API call complete...')

            if args.save_log_prob:
                choice = openai_response.get("choices", [{}])[0]
                logprobs_obj = choice.get("logprobs")
                token_ids = None
                token_ids_source = None
                if isinstance(logprobs_obj, dict):
                    token_ids = logprobs_obj.get("token_ids")
                    if token_ids is None and isinstance(logprobs_obj.get("content"), list):
                        token_ids = []
                        for item in logprobs_obj.get("content", []):
                            if not isinstance(item, dict):
                                token_ids.append(None)
                                continue
                            if item.get("token_id") is not None:
                                token_ids.append(item.get("token_id"))
                                token_ids_source = "vllm"
                                continue
                            token = item.get("token")
                            if token is None:
                                token_ids.append(None)
                                continue
                            tid = tokenizer.convert_tokens_to_ids(token)
                            if tid is not None and tid != tokenizer.unk_token_id:
                                token_ids.append(tid)
                                token_ids_source = "local_tokenizer"
                                continue
                            encoded = tokenizer.encode(token, add_special_tokens=False)
                            token_ids.append(encoded[0] if len(encoded) == 1 else None)
                            if len(encoded) == 1:
                                token_ids_source = "local_tokenizer"
                log_probs.append({
                    "iter": it,
                    "logprobs": logprobs_obj,
                    "token_ids": token_ids,
                    "token_ids_source": token_ids_source,
                    "prompt_logprobs": choice.get("prompt_logprobs"),
                })

            messages.append({'role': 'assistant', 'content': model_res})

            # try:
            #     assert 'Action:' in model_res
            # except AssertionError as e:
            #     logging.error(e)
            #     fail_obs = "Format ERROR: 'Action' must be included in your reply."
            #     continue

            # chosen_action = re.split(pattern, model_res)[2].strip()
            # print(model_res)
            # print(f'Chosen action: {chosen_action}')

            if "\"action\": \"answer\"" in model_res:
                break

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            try:
                if action_key in ['left_click','right_click','middle_click','double_click','triple_click']:
                    await exec_action_click({'x':coordinate_absolute[0],'y':coordinate_absolute[1]}, page)
                elif action_key == "mouse_move":
                    await page.mouse.move(coordinate_absolute[0], coordinate_absolute[1])
                elif action_key =='type':
                    await exec_action_type({'content': action['arguments']['text']}, page)
                elif action_key =='key':
                    await perform_hotkey(page, action['arguments']['keys'])
                elif action_key =='scroll': 
                    await exec_action_scroll(action['arguments']['pixels'], page)
                elif action_key =='hscroll': 
                    await exec_action_hscroll(action['arguments']['pixels'], page)
                elif action_key =='left_click_drag':
                    await exec_action_drag({'x1':coordinate_absolute[0],'y1':coordinate_absolute[1],'x2':coordinate_absolute[0],'y2':coordinate_absolute[1]}, page)
                elif action_key == 'wait':
                    # await asyncio.sleep(action['arguments']['time'])
                    await asyncio.sleep(10)
                elif action_key == 'goback':
                    await page.go_back()
                    await asyncio.sleep(3)
                elif action_key =='answer' or action_key =='terminate': 
                    break
                await asyncio.sleep(3)
            except Exception as e:
                logging.error(f"Exec error: {e}")
                fail_obs = "The action cannot be executed. Please revise."
                await asyncio.sleep(3)
                continue

        print_message(messages, task_dir)
        if args.save_log_prob:
            with open(os.path.join(task_dir, 'log_probs.json'), 'w', encoding='utf-8') as fw:
                json.dump(log_probs, fw, indent=2, ensure_ascii=False)
        task_logger.info(f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')
    except Exception as e:
        task_logger.exception(f"Task failed unexpectedly: {e}")
    finally:
        await safe_close(browser, pw, task_logger)
        task_logger.info("Task completed.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='/home/yifeihe/cua/WebVoyager/data/webvoyager_train_data_easy_sample_5120.jsonl')
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    parser.add_argument("--num_trials", type=int, default=1, help="Number of times to run each task.")
    parser.add_argument("--save_log_prob", action='store_true')
    parser.add_argument("--prompt_logprobs", type=int, default=0, help="Number of prompt logprobs to return (vLLM/OpenAI style).")
    parser.add_argument("--completion_logprobs_k", type=int, default=1, help="Top-k completion logprobs to return (vLLM expects int).")
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--model", type=str, default='gpt', choices=['gpt', 'uitars', 'qwen3'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=-1, help="Maximum number of tasks to run. If -1, run all tasks.")
    parser.add_argument("--exclude_tasks", type=str, default=["allrecipe", "github"], help="Tasks to exclude. Comma separated list of task ids.")
    parser.add_argument("--use_webgym_tasks", action='store_true')

    args = parser.parse_args()

    # current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    # log_file_path = os.path.join(os.path.join(args.output_dir, current_time), 'main.log')
    # logger = setup_main_logger(log_file_path)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # Dummy key to satisfy the client
    )

    # Save Result file
    # result_dir = os.path.join(args.output_dir, current_time)
    result_dir = os.path.join(args.output_dir, f'round_{args.round}')
    print(result_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Load tasks
    if args.use_webgym_tasks:
        raw_tasks = load_dataset("microsoft/webgym_tasks", split="train")
        tasks = [normalize_webgym_task(task) for task in raw_tasks]
    else:   
        tasks = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                tasks.append(json.loads(line))
    
    exclude_patterns = []
    print(f"Excluding tasks: {args.exclude_tasks}")
    for item in args.exclude_tasks:
        exclude_patterns.extend([p.strip().lower() for p in item.split(",") if p.strip()])

    if exclude_patterns:
        tasks = [
            task for task in tasks
            if not any(pattern in str(task.get("id", "")).lower() for pattern in exclude_patterns)
        ]


    if args.max_tasks > 0 and len(tasks) > args.max_tasks:
        tasks = random.sample(tasks, args.max_tasks)
    
    # Parallelize tasks using asyncio with a limit of 16 concurrent tasks
    # async def run_all_tasks():
    #     # Create a semaphore to limit concurrency to 16 tasks at a time
    #     semaphore = asyncio.Semaphore(8)
        
    #     async def run_task_with_semaphore(task_id):
    #         async with semaphore:
    #             await run_task(task_id, tasks[task_id], args, result_dir, options, client)
        
    #     # Create a task for each item and run them with the semaphore limit
    #     await asyncio.gather(*[
    #         run_task_with_semaphore(task_id)
    #         for task_id in range(len(tasks))
    #     ])

    # asyncio.run(run_all_tasks())
    
    # Vanilla way to parallelize tasks using asyncio
    # async def run_all_tasks():
    #     await asyncio.gather(*[
    #         run_task(task_id, tasks[task_id], args, result_dir, options, client)
    #         for task_id in range(len(tasks))
    #     ])
    # asyncio.run(run_all_tasks())

    # Use ThreadPoolExecutor for parallelism
    def run_task_sync(task_id, trial_id):
        asyncio.run(run_task(task_id, tasks[task_id], trial_id, args, result_dir, client))

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for task_id in range(len(tasks)):
            for trial_id in range(1, args.num_trials + 1):
                futures.append(executor.submit(run_task_sync, task_id, trial_id))
        for future in concurrent.futures.as_completed(futures):
            future.result()
        
        

if __name__ == '__main__':
    main()
    print('End of process')
