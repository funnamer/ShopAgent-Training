#!/usr/bin/env bash
set -euo pipefail

GRPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${GRPO_DIR}/.." && pwd)
SHOP_SIMULATOR_ROOT=${SHOP_SIMULATOR_ROOT:-${PROJECT_ROOT}/shopSimulator}
STORAGE_ROOT=${STORAGE_ROOT:-/root/autodl-tmp/ShopAgent-Training/rl}
RL_PYTHON=${RL_PYTHON:-/root/miniconda3/envs/rl/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/Qwen3-4B-sft150-merged}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
TRAIN_FILE=${TRAIN_FILE:-${STORAGE_ROOT}/data/train.parquet}
VAL_FILE=${VAL_FILE:-${STORAGE_ROOT}/data/val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${STORAGE_ROOT}/outputs/qwen3-4b-checkpoint150-grpo}

# 4 prompts x 4 samples = 16 concurrent sessions, below the 20-slot server.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
ENV_SLOTS=${ENV_SLOTS:-20}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-42}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-12000}
ACTOR_LR=${ACTOR_LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
# Use every GPU visible to the rl environment by default. Override this when a
# subset is selected with CUDA_VISIBLE_DEVICES or for a single-GPU debug run.
if [[ -z "${N_GPUS_PER_NODE:-}" && "${CONFIG_ONLY:-0}" == "1" ]]; then
  N_GPUS_PER_NODE=1
elif [[ -z "${N_GPUS_PER_NODE:-}" ]]; then
  N_GPUS_PER_NODE=$("${RL_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
fi
LOGGER=${LOGGER:-console}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlog}
SWANLAB_MODE=${SWANLAB_MODE:-cloud}

if (( MAX_MODEL_LEN <= MAX_PROMPT_LENGTH )); then
  echo "MAX_MODEL_LEN must be greater than MAX_PROMPT_LENGTH" >&2
  exit 2
fi

if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
  echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LEN" >&2
  exit 2
fi

if (( N_GPUS_PER_NODE < 1 )) && [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  echo "No CUDA GPU is visible to ${RL_PYTHON}" >&2
  exit 2
fi

if (( TRAIN_BATCH_SIZE * ROLLOUT_N > ENV_SLOTS )); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must not exceed ${ENV_SLOTS} ShopSimulator slots" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${SHOP_SIMULATOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=true
# Four colocated vLLM workers otherwise inherit 60 OpenMP threads each and
# contend heavily on this 128-thread host.
export OMP_NUM_THREADS=${GRPO_OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${GRPO_MKL_NUM_THREADS:-16}
export HYDRA_FULL_ERROR=1
export SWANLAB_LOG_DIR
export SWANLAB_MODE
export SHOPSIM_CATALOG_PATH="${SHOP_SIMULATOR_ROOT}/shop_env/data/fine_items_eval_train_all.json"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  "${RL_PYTHON}" -m rl.prepare_dataset
fi

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  "${RL_PYTHON}" -m rl.preflight \
    --model "${MODEL_PATH}" \
    --train "${TRAIN_FILE}" \
    --val "${VAL_FILE}"
fi

ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size=4
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.return_raw_chat=True
  data.filter_overlong_prompts=False
  data.truncation=error
  data.dataloader_num_workers=2
  +data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_shm=True
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  actor_rollout_ref.model.target_modules=all-linear
  # vLLM supports verl's native adapter sync. Keeping the adapter separate
  # avoids a full merged-weight refit and preserves Qwen3's tied
  # embed_tokens/lm_head alias exactly as it was loaded from safetensors.
  actor_rollout_ref.model.lora.merge=False
  # Keep the padded SDPA path: this environment intentionally does not require
  # the ABI-sensitive flash-attn package.
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.fsdp_config.offload_policy=True
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  # Never let vLLM fall back to Qwen3's native 40,960-token context here: on
  # a colocated single-GPU run that needlessly reserves several GiB of KV.
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  # Qwen's documented non-thinking sampling defaults.
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=0.8
  actor_rollout_ref.rollout.top_k=20
  actor_rollout_ref.rollout.load_format=safetensors
  # Official verl recommendation for FSDP/FSDP2 LoRA on GPUs below 48 GiB.
  actor_rollout_ref.rollout.layered_summon=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.max_num_seqs=16
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_ASSISTANT_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${GRPO_DIR}/configs/tools.yaml"
  actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable
  actor_rollout_ref.rollout.agent.default_agent_loop=shop_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${GRPO_DIR}/configs/agent_loop.yaml"
  actor_rollout_ref.rollout.agent.num_workers=4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}"
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  reward.num_workers=2
  reward.custom_reward_function.path="${GRPO_DIR}/reward.py"
  reward.custom_reward_function.name=compute_score
  trainer.project_name=shopsimulator_grpo
  trainer.experiment_name=qwen3-4b-checkpoint150-lora
  # Use verl's supported synchronous trainer with async multi-turn rollout.
  trainer.use_v1=False
  trainer.logger="[\"${LOGGER}\",\"swanlab\"]"
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
  trainer.nnodes=1
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.val_before_train=False
  trainer.default_local_dir="${OUTPUT_DIR}"
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts"
  trainer.validation_data_dir="${OUTPUT_DIR}/validation"
  trainer.critic_warmup=0
  +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=rl.metrics.install
)

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exec "${RL_PYTHON}" -m verl.trainer.main_ppo "${ARGS[@]}" --cfg job
fi

exec "${RL_PYTHON}" -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
