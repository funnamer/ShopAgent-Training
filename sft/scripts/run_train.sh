#!/usr/bin/env bash
set -euo pipefail

cd /root/ShopAgent-Training/sft
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec conda run --no-capture-output -n ft \
  python train_sft.py \
  --config configs/qwen3_4b_action_only_lora.json \
  "$@"
