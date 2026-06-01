# PRO-CUA

This is the official codebase for **PRO-CUA: Process-Reward Optimization for Computer Use Agents**. It contains the browser-agent inference/data-collection stack, SFT baselines, and GRPO/Step-RL training recipes used for Qwen3-VL 4B and 8B computer-use agents.

## Repository Layout

- `WebVoyager/`: browser-agent inference and data collection. It launches vLLM servers, routes model requests, drives Playwright/WebDriver interactions, and evaluates/labels trajectories. The task/query files live under `WebVoyager/data/` and include WebVoyager, Online-Mind2Web, and related browser-use datasets.
- `verl/`: RL training code based on verl. The PRO-CUA entry points run GRPO-style training with either rule-based rewards or process-reward-model rewards.
- `LLaMA-Factory/`: supervised fine-tuning baseline code used by the FBC entry points.

Large generated artifacts are intentionally not part of the release. Runtime outputs default to repo-local `data/`, `ckpts/`, `tmp/`, and `.cache/` directories unless overridden.

## Main Training Entry Points

The six main scripts are:

| Method | 4B | 8B |
| --- | --- | --- |
| FBC / SFT baseline | `WebVoyager/scripts/fbc_4b.sh` | `WebVoyager/scripts/fbc_8b.sh` |
| Step-RL, rule-based reward | `verl/scripts/step_rl_rule_based_reward_4b.sh` | `verl/scripts/step_rl_rule_based_reward_8b.sh` |
| PRO-CUA, process reward | `verl/scripts/pro_cua_4b.sh` | `verl/scripts/pro_cua_8b.sh` |

Each script checks whether the expected `round_1` rollout data exists. If it is missing, the script calls `WebVoyager/scripts/webvoyager_collect_data.sh` to collect and label initial trajectories before training.

## Environment Variables

Set these before running jobs:

```bash
export PRO_CUA_ROOT=/path/to/PRO-CUA
export PRO_CUA_DATA_ROOT=$PRO_CUA_ROOT/data
export PRO_CUA_CKPT_ROOT=$PRO_CUA_ROOT/ckpts

export HF_TOKEN=YOUR_HUGGINGFACE_TOKEN          # optional, for gated HF models
export WANDB_API_KEY=YOUR_WANDB_API_KEY        # optional, for W&B logging
export MODEL_API_KEY=YOUR_MODEL_ROUTER_KEY     # optional, passed to WebVoyager model calls
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY      # used by auto-evaluation/labeling
```

For FBC/SFT jobs also set:

```bash
export LLAMAFACTORY_ROOT=$PRO_CUA_ROOT/LLaMA-Factory
```

Useful overrides:

```bash
export WEBVOYAGER_ROOT=$PRO_CUA_ROOT/WebVoyager
export VERL_ROOT=$PRO_CUA_ROOT/verl
export VLLM_BOOT_WAIT_SECONDS=120
export HF_CACHE_DIR=$PRO_CUA_ROOT/.cache/huggingface
export VLLM_DOWNLOAD_DIR=$PRO_CUA_ROOT/.cache/vllm
```

## Environments and Dependencies

The scripts expect three conda environments:

### `verl-agent`

Used for browser interaction, data collection, trajectory construction, vLLM serving, and Qwen-based auto-evaluation.

Suggested packages:

```bash
conda create -n verl-agent python=3.10 -y
conda activate verl-agent
pip install torch torchvision transformers datasets accelerate openai flask requests pillow numpy pandas tqdm qwen-vl-utils vllm
pip install playwright
python -m playwright install chromium
```

If your system does not already have browser runtime libraries, install Playwright's OS dependencies with the command recommended by Playwright for your platform.

### `verl`

Used by the Step-RL and PRO-CUA scripts for GRPO training, rollout, reward computation, Ray, and checkpoint merging.

```bash
conda create -n verl python=3.10 -y
conda activate verl
cd $PRO_CUA_ROOT/verl
pip install -e .
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

The current scripts use FSDP2, Ray, vLLM rollout, `wandb`, `flash-attn`, and Qwen3-VL models. Match CUDA, PyTorch, vLLM, and flash-attn versions to your cluster.

### `llamafactory`

Used only by `WebVoyager/scripts/fbc_4b.sh` and `WebVoyager/scripts/fbc_8b.sh`.

```bash
conda create -n llamafactory python=3.10 -y
conda activate llamafactory
cd $LLAMAFACTORY_ROOT
pip install -e .
pip install -r requirements.txt
```

The FBC scripts call `llamafactory-cli train examples/train_full/qwen3vl_full_sft.yaml`.

## Running Jobs

From the repository root:

```bash
# FBC / SFT
bash WebVoyager/scripts/fbc_4b.sh
bash WebVoyager/scripts/fbc_8b.sh

# Step-RL with rule-based reward
bash verl/scripts/step_rl_rule_based_reward_4b.sh
bash verl/scripts/step_rl_rule_based_reward_8b.sh

# PRO-CUA with process reward
bash verl/scripts/pro_cua_4b.sh
bash verl/scripts/pro_cua_8b.sh
```

The scripts assume an 8-GPU node by default. Adjust GPU counts, batch sizes, vLLM ports, and memory settings inside the scripts for other hardware.

## WebVoyager Inference Stack

`WebVoyager/scripts/launch_vllm_servers.sh` launches one vLLM OpenAI-compatible server per GPU, starting at port `8001` by default. `flask_router.py` routes requests across those local servers. The browser agent then uses Playwright to interact with webpages for tasks in `WebVoyager/data/`.

Useful launcher overrides:

```bash
export NUM_PORTS=8
export BASE_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=12800
```

## Notes

- No personal absolute paths or tokens are required by the main scripts. Paths are derived from `PRO_CUA_ROOT` and can be overridden with environment variables.
- The scripts write checkpoints under `${PRO_CUA_CKPT_ROOT}` and rollouts/evaluation data under `${PRO_CUA_DATA_ROOT}`.
- Auto-evaluation requires an OpenAI-compatible API key in `OPENAI_API_KEY`.

## Citation
```
@article{he2026pro,
  title={PRO-CUA: Process-Reward Optimization for Computer Use Agents},
  author={He, Yifei and Yang, Rui and Bai, Hao and Zhang, Tong and Zhao, Han},
  journal={arXiv preprint arXiv:2605.29119},
  year={2026}
}
```