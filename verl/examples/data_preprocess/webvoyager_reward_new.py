import json
import aiohttp
import os
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import re
import base64
import asyncio
import random
import sys
from io import BytesIO
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

from data_visualization import process_single_image


PRM_PROMPT = """You are an expert evaluator grading a Computer-Use Agent. Your role is to evaluate whether the agent’s proposed next action is the strictly correct and necessary step to advance the given task.

You are provided with:
1. The overarching task instruction.
2. The history of actions taken so far.
3. The CURRENT screenshot (the state immediately BEFORE the proposed action), annotated to show the proposed target of the action.
4. The proposed Thought and Action Code.

The screenshot is an annotated visualization of the proposed action, not a raw screenshot:
- Red marks, arrows, or points indicate where the proposed action is targeting.
- Small index labels and overlay text are part of the annotation.
- Use these annotations to judge whether the proposed action is correctly grounded on the UI.
- Do not confuse the annotation itself with a native page element.

<task_instruction>
{instruction}
</task_instruction>

<history_actions>
{history_actions}
</history_actions>

<proposed_action>
Step {step_index}: {action_code}
</proposed_action>

### Evaluation Criteria
You must evaluate the proposed action and output a binary decision: is the action CORRECT or INCORRECT?

An action is INCORRECT if it exhibits ANY of the following flaws:
- Grounding Failure: The code targets the wrong coordinates, a non-existent element, or the wrong input field based on the provided screenshot.
- Hallucination: The agent assumes a state that is not visually present.
- Inefficiency/Redundancy: The action needlessly repeats a past step from the history, performs useless scrolling, or wastes a step without advancing the task.
- Logical Progression Failure: The action executes successfully but does not move the agent closer to the final goal.

An action is CORRECT ONLY if it is visually grounded, mathematically accurate, and actively advances the task toward completion.

### Output Format
Provide a rigorous step-by-step reflection. You must perform a "mental rollout" to predict the consequences of the action before determining if it facilitates task completion. Then, output a strictly valid JSON block.

<analysis_process>
1. [Current State Assessment]: What is currently visible on the screen? What is the immediate blocker to completing the task?
2. [Target Verification]: Does the proposed code correctly and accurately target the intended UI element in the screenshot?
3. [Mental Rollout]: If this exact code is executed, what will happen? (e.g., "A dropdown menu will appear," "The page will scroll down," "The text 'shoes' will be typed").
4. [Task Alignment]: Does this predicted outcome meaningfully and efficiently advance the task? Or is it a redundant/wasteful action given the history?
5. [Final Verdict]: Conclude whether the step is Correct or Incorrect.
</analysis_process>

```json
{
  "is_correct": boolean,
  "reflection": "A 1-2 sentence summary of why the action was marked correct or incorrect."
}"""

MAX_PRM_RETRIES = 5
MIN_PRM_RETRY_DELAY_SECONDS = 3.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    try:
        gt_action = json.loads(ground_truth.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        gt_action_key = gt_action['arguments']['action']
        action = json.loads(solution_str.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        action_key = action['arguments']['action']
        
        # action key reward
        if action_key != gt_action_key:
            return 0.0
        
        # grounding reward
        if 'coordinate' in action['arguments'] and 'coordinate' in gt_action['arguments']:
            coordinate = action['arguments']['coordinate']
            gt_coordinate = gt_action['arguments']['coordinate']

            dx = abs(coordinate[0] - gt_coordinate[0])
            dy = abs(coordinate[1] - gt_coordinate[1])
            if dx > 10 or dy > 10:
                return 0.0
        
        # text reward
        if 'text' in action['arguments'] and 'text' in gt_action['arguments']:
            text = action['arguments']['text']
            gt_text = gt_action['arguments']['text']
            text_tokens = set(str(text).lower().split())
            gt_text_tokens = set(str(gt_text).lower().split())

            if len(text_tokens) == 0 and len(gt_text_tokens) == 0:
                f1 = 1.0
            elif len(text_tokens) == 0 or len(gt_text_tokens) == 0:
                f1 = 0.0
            else:
                common = len(text_tokens & gt_text_tokens)
                precision = common / len(text_tokens)
                recall = common / len(gt_text_tokens)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

            if f1 < 0.5:
                return 0.0
        
        return 1

    except Exception as e:
        print(f"Error when parsing model response: {e}")
        return 0.0


def _extract_score_from_content(content):
    """
    Parses the binary reward score from the PRM's generated content.
    Expects a JSON block containing "is_correct": true/false.
    Returns 1 for Correct, 0 for Incorrect, and None if unparseable.
    """
    # Regex to find the is_correct boolean value.
    # The (?:"|\')? makes it robust to missing or alternate quotes around the key.
    # The (true|false) captures the boolean, ignoring case.
    correct_match = re.search(
        r'(?:"|\')?is_correct(?:"|\')?\s*:\s*(true|false)',
        content,
        re.IGNORECASE
    )
    
    if correct_match:
        is_correct = correct_match.group(1).lower() == 'true'
        
        # Return 1 (positive advantage) for correct, 0 (negative advantage) for incorrect
        return 1 if is_correct else 0

    # Optional: Add a fallback print/logging statement here in your actual codebase 
    # to catch and debug LLM outputs that completely failed to follow the schema.
    return None


def _extract_json_block(content):
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)

    inline_match = re.search(r"(\{.*\})", content, re.DOTALL)
    if inline_match:
        return inline_match.group(1)

    return None


def _extract_prm_metadata(content):
    json_block = _extract_json_block(content)
    if not json_block:
        return {
            "prm_is_correct": None,
            "prm_reflection": "",
            "prm_parse_error": "No JSON block found in PRM response",
        }

    try:
        parsed = json.loads(json_block)
        is_correct = parsed.get("is_correct")
        if is_correct is None:
            normalized_is_correct = None
        else:
            normalized_is_correct = int(bool(is_correct))
        return {
            "prm_is_correct": normalized_is_correct,
            "prm_reflection": str(parsed.get("reflection", "")),
            "prm_parse_error": "No error",
        }
    except Exception as e:
        return {
            "prm_is_correct": None,
            "prm_reflection": "",
            "prm_parse_error": str(e),
        }


def _empty_prm_metadata(parse_error="No PRM metadata"):
    return {
        "prm_is_correct": None,
        "prm_reflection": "",
        "prm_parse_error": parse_error,
    }


def _should_retry_prm_error(exc):
    error_text = str(exc).lower()
    retry_markers = [
        "rate limit",
        "rate_limit_exceeded",
        "429",
        "too many requests",
    ]
    return any(marker in error_text for marker in retry_markers)

# def _extract_score_from_content(content):
#     score_patterns = [
#         r'Expected value:\s*(\d+)',
#         r'Expected value:\s*\n\s*(\d+)',
#     ]
#     for pattern in score_patterns:
#         match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
#         if match:
#             score = int(match.group(1))
#             if 0 <= score <= 10:
#                 return score

#     float_patterns = [
#         r'Expected value:\s*(-?\d+(?:\.\d+)?)',
#         r'Expected value:\s*\n\s*(-?\d+(?:\.\d+)?)',
#     ]
#     for pattern in float_patterns:
#         match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
#         if match:
#             val = float(match.group(1))
#             if -1.0 <= val <= 1.0:
#                 return max(0, min(10, round((val + 1) * 5)))

#     correct_match = re.search(
#         r'last_step_correct"\s*:\s*(true|false)',
#         content,
#         re.IGNORECASE
#     )
#     redundant_match = re.search(
#         r'last_step_redundant"\s*:\s*(true|false)',
#         content,
#         re.IGNORECASE
#     )
#     if correct_match:
#         is_correct = correct_match.group(1).lower() == 'true'
#         is_redundant = redundant_match and redundant_match.group(1).lower() == 'true'
#         if is_correct and not is_redundant:
#             return 1
#         if is_correct and is_redundant:
#             return 0.5
#         return 0

#     return None
    

async def chat_complete(router_address: str, chat_complete_request: dict):
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=chat_complete_request) as resp:
            resp.raise_for_status()
            output = await resp.json()
            return ChatCompletion(**output)


def _build_prm_messages(solution_str, extra_info):
    user_prompt_tmp = PRM_PROMPT.replace('{instruction}', extra_info["task_instruction"])
    user_prompt_tmp = user_prompt_tmp.replace('{step_index}', str(extra_info["gold_turn_index"]))
    user_prompt_tmp = user_prompt_tmp.replace('{action_code}', solution_str)
    user_prompt_tmp = user_prompt_tmp.replace('{history_actions}', json.dumps(extra_info["history_actions"]))

    with Image.open(extra_info["image_path"]) as image:
        try:
            annotated_img, _ = process_single_image(
                solution_str,
                image,
                ins_cmd=extra_info["task_instruction"],
            )
        except Exception:
            annotated_img = image.copy()


    buffer = BytesIO()
    annotated_img.save(buffer, format="PNG")
    b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_tmp},
                {"type": "text", "text": "Screenshot BEFORE last action"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": "Your judgement:\n"},
            ],
        }
    ]


def _format_prm_result(content, retry_count=0):
    score = _extract_score_from_content(content)
    metadata = _extract_prm_metadata(content)
    if score is None:
        return {
            "score": 0.0,
            "prm_response": content,
            "prm_error": "No score found in the response",
            "prm_retry_count": retry_count,
            **metadata,
        }

    return {
        "score": score,
        "prm_response": content,
        "prm_error": "No error",
        "prm_retry_count": retry_count,
        **metadata,
    }

async def compute_score_prm(data_source, solution_str, ground_truth, reward_router_address, reward_model_tokenizer, extra_info=None, **kwargs):
    messages = _build_prm_messages(solution_str, extra_info)

    request = {
        "messages": messages,
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "temperature": 0
    }

    for attempt_idx in range(MAX_PRM_RETRIES):
        try:
            result = await chat_complete(
                router_address=reward_router_address,
                chat_complete_request=request,
            )

            content = result.choices[0].message.content.strip()
            return _format_prm_result(content, retry_count=attempt_idx)

        except Exception as e:
            if attempt_idx < MAX_PRM_RETRIES - 1 and _should_retry_prm_error(e):
                await asyncio.sleep(3)
                continue

            return {
                "score": 0.0,
                "prm_response": "",
                "prm_error": str(e),
                "prm_retry_count": attempt_idx,
                **_empty_prm_metadata(parse_error=str(e)),
            }


async def compute_score_prm_gpt(data_source, solution_str, ground_truth, reward_router_address=None, reward_model_tokenizer=None, extra_info=None, **kwargs):
    messages = _build_prm_messages(solution_str, extra_info)

    openai_base_url = kwargs.get("openai_base_url")

    client_kwargs = {"api_key": "<YOUR-OPENAI-API-KEY"}
    if openai_base_url:
        client_kwargs["base_url"] = openai_base_url

    client = AsyncOpenAI(**client_kwargs)
    try:
        model_names = ["gpt-5-mini", "gpt-5.4-nano"]
        last_error = None

        for model_idx, model_name in enumerate(model_names):
            for attempt_idx in range(MAX_PRM_RETRIES):
                try:
                    result = await client.chat.completions.create(
                        model=model_name,
                        messages=messages
                    )

                    content = result.choices[0].message.content.strip()
                    total_retry_count = model_idx * MAX_PRM_RETRIES + attempt_idx
                    formatted = _format_prm_result(content, retry_count=total_retry_count)
                    formatted["prm_model"] = model_name
                    return formatted

                except Exception as e:
                    last_error = e
                    is_retryable = _should_retry_prm_error(e)
                    has_more_retries_for_model = attempt_idx < MAX_PRM_RETRIES - 1
                    has_fallback_model = model_idx < len(model_names) - 1

                    if is_retryable and has_more_retries_for_model:
                        await asyncio.sleep(MIN_PRM_RETRY_DELAY_SECONDS)
                        continue

                    if is_retryable and not has_more_retries_for_model and has_fallback_model:
                        break

                    total_retry_count = model_idx * MAX_PRM_RETRIES + attempt_idx
                    return {
                        "score": 0.0,
                        "prm_response": "",
                        "prm_error": str(e),
                        "prm_retry_count": total_retry_count,
                        "prm_model": model_name,
                        **_empty_prm_metadata(parse_error=str(e)),
                    }

        total_retry_count = len(model_names) * MAX_PRM_RETRIES - 1
        return {
            "score": 0.0,
            "prm_response": "",
            "prm_error": str(last_error) if last_error is not None else "Unknown PRM error",
            "prm_retry_count": total_retry_count,
            "prm_model": model_names[-1],
            **_empty_prm_metadata(parse_error=str(last_error) if last_error is not None else "Unknown PRM error"),
        }
    finally:
        await client.close()

async def compute_score_random(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return random.randint(0, 1)


# async def main():
#     data_source = "webvoyager"
#     solution_str = "I will open the browser and search for the answer."
#     ground_truth = "I will open the browser and search for the answer."
#     reward_router_address = "localhost:8000"
#     reward_model_tokenizer = None
#     extra_info = {'gold_turn_index': 4, 'history_actions': [{'arguments': {'action': 'left_click', 'coordinate': [42, 33], 'keys': None, 'pixels': None, 'status': None, 'text': None}, 'name': 'computer_use'}, {'arguments': {'action': 'left_click', 'coordinate': [175, 447], 'keys': None, 'pixels': None, 'status': None, 'text': None}, 'name': 'computer_use'}, {'arguments': {'action': 'type', 'coordinate': None, 'keys': None, 'pixels': None, 'status': None, 'text': 'Project-H language:Django'}, 'name': 'computer_use'}, {'arguments': {'action': 'left_click', 'coordinate': [867, 96], 'keys': None, 'pixels': None, 'status': None, 'text': None}, 'name': 'computer_use'}], 'image_path': '/data/common/yifei/cua/data/qwen3_WebVoyager_data_clean/sft_qwen3_4b_512tasks_30steps_correct_traj/round_2/taskAmazon--0-1/screenshot4.png', 'index': 10, 'num_turns': 8, 'task_instruction': "Now given a task: Find the repository 'Project-H' that has the most forks in the 'Django' language, and compare its features with a similar repository in the 'Flask' language."}
#     kwargs = {}
#     result = await compute_score_prm_gpt(data_source, solution_str, ground_truth, reward_router_address, reward_model_tokenizer, extra_info, **kwargs)
#     print(result)

# asyncio.run(main())
