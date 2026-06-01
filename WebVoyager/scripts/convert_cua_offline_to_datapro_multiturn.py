#!/usr/bin/env python3
"""
Convert offline CUA rollouts into trajectory-level DataProto samples.

Each task becomes one sample:
- prompt: prefix ending at the first user observation/image
- response: the remaining multi-turn continuation (assistant and later user turns)
- response_mask: 1 only on assistant-token spans, 0 elsewhere
- rm_scores: terminal reward on the last assistant token

This keeps causal ordering across turns so PPO/GAE can propagate credit within a trajectory sample.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import hf_processor, hf_tokenizer


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_user_obs_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Collect per-step observation texts from structured user messages."""
    obs_texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        text = None
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text = item["text"].strip()
                break
        if text:
            obs_texts.append(text)
    return obs_texts


def extract_logprobs(lp_item: dict[str, Any]) -> list[float]:
    content = lp_item.get("logprobs", {}).get("content", [])
    return [float(tok.get("logprob", 0.0)) for tok in content]


def build_mm_inputs(prompt_inputs):
    if hasattr(prompt_inputs, "convert_to_tensors"):
        mm = dict(prompt_inputs.convert_to_tensors("pt"))
    else:
        mm = dict(prompt_inputs)
    mm.pop("input_ids", None)
    mm.pop("attention_mask", None)

    image_grid_thw = mm.get("image_grid_thw")
    if image_grid_thw is not None:
        images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
        mm["images_seqlens"] = images_seqlens
        pixel_values = mm.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            num_tokens = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())
            if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)
            if pixel_values.ndim == 2:
                if pixel_values.shape[0] == num_tokens:
                    mm["pixel_values"] = pixel_values
                elif pixel_values.shape[1] == num_tokens:
                    mm["pixel_values"] = pixel_values.transpose(0, 1).contiguous()
                else:
                    raise ValueError(
                        f"pixel_values shape {tuple(pixel_values.shape)} does not match num_tokens={num_tokens}"
                    )
            else:
                raise ValueError(f"Unsupported pixel_values shape: {tuple(pixel_values.shape)}")
    return mm


def compute_position_ids(processor, attention_mask, input_ids, multi_modal_inputs):
    if processor is None or not hasattr(processor, "get_rope_index"):
        return compute_position_id_with_mask(attention_mask)
    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    video_grid_thw = multi_modal_inputs.get("video_grid_thw")
    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)
    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
    text_position_ids = text_position_ids.unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=1)


def pad_prompt_response(tokenizer, prompt_ids, response_ids, response_mask, prompt_len, response_len):
    tokenizer.padding_side = "left"
    prompt_output = tokenizer.pad(
        {"input_ids": prompt_ids},
        padding="max_length",
        max_length=prompt_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if prompt_output["input_ids"].dim() == 1:
        prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
        prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

    tokenizer.padding_side = "right"
    response_output = tokenizer.pad(
        {"input_ids": response_ids},
        padding="max_length",
        max_length=response_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if response_output["input_ids"].dim() == 1:
        response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
        response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

    response_mask_output = tokenizer.pad(
        {"input_ids": response_mask},
        padding="max_length",
        max_length=response_len,
        return_tensors="pt",
        return_attention_mask=False,
    )
    if response_mask_output["input_ids"].dim() == 1:
        response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)
    response_mask_tensor = response_mask_output["input_ids"] * response_output["attention_mask"]

    attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
    input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

    return (
        prompt_output["input_ids"],
        response_output["input_ids"],
        response_mask_tensor,
        attention_mask,
        input_ids,
    )


def tokenize_with_images(tokenizer, processor, messages: list[dict[str, Any]], images: list[Image.Image]):
    if processor is not None and hasattr(processor, "apply_chat_template"):
        prompt_str = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    else:
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return processor(
        text=[prompt_str],
        images=images if images else None,
        return_tensors="pt",
        do_sample_frames=False,
    )


def convert_task_dirs(task_dirs: list[Path], results: dict[str, Any], tokenizer, processor, args):
    all_prompts = []
    all_responses = []
    all_response_masks = []
    all_attention_masks = []
    all_input_ids = []
    all_position_ids = []
    all_rollout_log_probs = []
    all_rm_scores = []
    all_multi_modal_inputs = []
    all_num_turns = []

    for task_dir in task_dirs:
        interact_path = task_dir / "interact_messages.json"
        logprobs_path = task_dir / "log_probs.json"
        if not interact_path.exists() or not logprobs_path.exists():
            continue

        messages_raw = load_json(interact_path)
        logprobs = load_json(logprobs_path)
        system_msgs = [m for m in messages_raw if m.get("role") == "system"]
        user_text_msgs = [m for m in messages_raw if m.get("role") == "user" and isinstance(m.get("content"), str)]
        assistant_msgs = [m.get("content", "") for m in messages_raw if m.get("role") == "assistant"]
        obs_texts = extract_user_obs_texts(messages_raw)
        if not user_text_msgs or not assistant_msgs or not obs_texts:
            continue

        initial_user = user_text_msgs[0]
        total_steps = min(len(assistant_msgs), len(logprobs), len(obs_texts))
        if total_steps <= 0:
            continue

        screenshots: list[Image.Image] = []
        for i in range(1, total_steps + 1):
            img_path = task_dir / f"screenshot{i}.png"
            if not img_path.exists():
                break
            screenshots.append(Image.open(img_path).convert("RGB"))
        total_steps = min(total_steps, len(screenshots))
        if total_steps <= 0:
            continue

        # Build full trajectory conversation with causal alternation.
        full_msgs: list[dict[str, Any]] = []
        full_msgs.extend(system_msgs)
        full_msgs.append(initial_user)
        for i in range(total_steps):
            full_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": obs_texts[i]},
                        {"type": "image"},
                    ],
                }
            )
            full_msgs.append({"role": "assistant", "content": assistant_msgs[i]})

        # Prompt prefix is up to first user observation (turn 1) only.
        prompt_msgs = system_msgs + [initial_user] + [full_msgs[len(system_msgs) + 1]]  # first step user obs/image

        full_inputs = tokenize_with_images(tokenizer, processor, full_msgs, screenshots[:total_steps])
        prompt_inputs = tokenize_with_images(tokenizer, processor, prompt_msgs, screenshots[:1])
        full_ids = full_inputs["input_ids"][0]
        prompt_ids = prompt_inputs["input_ids"][0]

        if full_ids.numel() <= prompt_ids.numel():
            continue
        if not torch.equal(full_ids[: prompt_ids.numel()], prompt_ids):
            # Template mismatch; skip unsafe sample.
            continue

        response_ids = full_ids[prompt_ids.numel() :].tolist()
        if len(prompt_ids) > args.prompt_length or len(response_ids) > args.response_length:
            # Avoid truncation-induced image token/feature mismatch.
            continue

        # Build assistant-token mask and rollout_log_probs over full sequence.
        full_response_mask = torch.zeros_like(full_ids, dtype=torch.long)
        full_rollout_lp = torch.zeros_like(full_ids, dtype=torch.float32)

        cursor_msgs = system_msgs + [initial_user]
        used_images = 0
        for step in range(total_steps):
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": obs_texts[step]},
                    {"type": "image"},
                ],
            }
            before_assistant_msgs = cursor_msgs + [user_turn]
            after_assistant_msgs = before_assistant_msgs + [{"role": "assistant", "content": assistant_msgs[step]}]

            used_images += 1
            ids_before = tokenize_with_images(tokenizer, processor, before_assistant_msgs, screenshots[:used_images])[
                "input_ids"
            ][0]
            ids_after = tokenize_with_images(tokenizer, processor, after_assistant_msgs, screenshots[:used_images])[
                "input_ids"
            ][0]
            start = ids_before.numel()
            end = ids_after.numel()
            if end > start:
                full_response_mask[start:end] = 1
                lp_vals = extract_logprobs(logprobs[step])[: end - start]
                if lp_vals:
                    full_rollout_lp[start : start + len(lp_vals)] = torch.tensor(lp_vals, dtype=torch.float32)
            cursor_msgs = after_assistant_msgs

        response_mask = full_response_mask[prompt_ids.numel() :].tolist()
        rollout_log_probs = full_rollout_lp[prompt_ids.numel() :].tolist()

        (
            prompts_padded,
            responses_padded,
            response_mask_tensor,
            attention_mask,
            input_ids,
        ) = pad_prompt_response(
            tokenizer,
            prompt_ids.tolist(),
            response_ids,
            response_mask,
            args.prompt_length,
            args.response_length,
        )

        multi_modal_inputs = build_mm_inputs(full_inputs)
        position_ids = compute_position_ids(processor, attention_mask, input_ids, multi_modal_inputs)

        rollout_log_probs = rollout_log_probs[: args.response_length] + [0.0] * max(
            0, args.response_length - len(rollout_log_probs)
        )
        rollout_log_probs_tensor = torch.tensor(rollout_log_probs, dtype=torch.float32).unsqueeze(0)

        rm_scores = torch.zeros_like(response_mask_tensor, dtype=torch.float32)
        reward = float(results.get(task_dir.name, 0.0))
        nz = torch.where(response_mask_tensor[0] > 0)[0]
        if len(nz) > 0:
            rm_scores[0, int(nz[-1].item())] = reward

        all_prompts.append(prompts_padded)
        all_responses.append(responses_padded)
        all_response_masks.append(response_mask_tensor)
        all_attention_masks.append(attention_mask)
        all_input_ids.append(input_ids)
        all_position_ids.append(position_ids)
        all_rollout_log_probs.append(rollout_log_probs_tensor)
        all_rm_scores.append(rm_scores)
        all_multi_modal_inputs.append(multi_modal_inputs)
        all_num_turns.append(total_steps)

    if not all_prompts:
        return None

    if args.pad_to_multiple and args.pad_to_multiple > 1:
        remainder = len(all_prompts) % args.pad_to_multiple
        if remainder != 0:
            pad_count = args.pad_to_multiple - remainder
            for _ in range(pad_count):
                all_prompts.append(all_prompts[-1])
                all_responses.append(all_responses[-1])
                all_response_masks.append(all_response_masks[-1])
                all_attention_masks.append(all_attention_masks[-1])
                all_input_ids.append(all_input_ids[-1])
                all_position_ids.append(all_position_ids[-1])
                all_rollout_log_probs.append(all_rollout_log_probs[-1])
                all_rm_scores.append(all_rm_scores[-1])
                all_multi_modal_inputs.append(all_multi_modal_inputs[-1])
                all_num_turns.append(all_num_turns[-1])

    batch = TensorDict(
        {
            "prompts": torch.cat(all_prompts, dim=0),
            "responses": torch.cat(all_responses, dim=0),
            "response_mask": torch.cat(all_response_masks, dim=0),
            "input_ids": torch.cat(all_input_ids, dim=0),
            "attention_mask": torch.cat(all_attention_masks, dim=0),
            "position_ids": torch.cat(all_position_ids, dim=0),
            "rollout_log_probs": torch.cat(all_rollout_log_probs, dim=0),
            "rm_scores": torch.cat(all_rm_scores, dim=0),
        },
        batch_size=len(all_prompts),
    )
    non_tensor_batch = {
        "__num_turns__": np.array(all_num_turns, dtype=np.int32),
        "multi_modal_inputs": np.array(all_multi_modal_inputs, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


def main():
    parser = argparse.ArgumentParser(description="Convert CUA offline data to trajectory-level DataProto")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt_length", type=int, required=True)
    parser.add_argument("--response_length", type=int, required=True)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--chunk_tasks", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    results = load_json(Path(args.results_json))
    tokenizer = hf_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    processor = hf_processor(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)

    task_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if args.max_tasks > 0:
        task_dirs = task_dirs[: args.max_tasks]

    if args.chunk_tasks and args.chunk_tasks > 0:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        total_samples = 0
        for chunk_start in range(0, len(task_dirs), args.chunk_tasks):
            chunk = task_dirs[chunk_start : chunk_start + args.chunk_tasks]
            data = convert_task_dirs(chunk, results, tokenizer, processor, args)
            if data is None:
                continue
            chunk_end = chunk_start + len(chunk) - 1
            out_path = out_dir / f"part_{chunk_start:06d}_{chunk_end:06d}.datapro"
            data.save_to_disk(out_path)
            n = len(data.batch)
            total_samples += n
            print(f"Saved DataProto with {n} samples to {out_path}")
        print(f"Total samples across shards: {total_samples}")
        return

    data = convert_task_dirs(task_dirs, results, tokenizer, processor, args)
    if data is None:
        raise RuntimeError("No samples were converted. Check inputs and mappings.")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(out_path)
    n = len(data.batch)
    print(f"Saved DataProto with {n} samples to {out_path}")


if __name__ == "__main__":
    main()
