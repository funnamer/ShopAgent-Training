#!/usr/bin/env bash
set -euo pipefail

GRPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# One small GRPO group per visible GPU, two sampled trajectories per prompt,
# and one optimizer step. The short context is only for plumbing validation.
RL_PYTHON=${RL_PYTHON:-/root/miniconda3/envs/rl/bin/python}
if [[ -z "${N_GPUS_PER_NODE:-}" ]]; then
  N_GPUS_PER_NODE=$("${RL_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
fi
export RL_PYTHON
export N_GPUS_PER_NODE
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${N_GPUS_PER_NODE}}
export ROLLOUT_N=${ROLLOUT_N:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${N_GPUS_PER_NODE}}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-23552}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-42}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2500}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/ShopAgent-Training/rl/outputs/smoke}

exec bash "${GRPO_DIR}/run_grpo.sh" trainer.total_training_steps=1 "$@"
