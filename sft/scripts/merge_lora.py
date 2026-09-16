#!/usr/bin/env python
"""Materialize a PEFT LoRA checkpoint as a standalone Transformers model.

Run this in the fine-tuning (``ft``) environment.  The resulting directory
contains the base model weights with the adapter merged into them, so downstream
RL/verl jobs do not need PEFT or the SFT checkpoint at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="checkpoint-N directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_model).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"base model does not exist: {base}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter checkpoint is invalid: {adapter}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is non-empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading base model: {base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"Loading adapter: {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    print("Merging LoRA weights...", flush=True)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(str(output), safe_serialization=True, max_shard_size=args.max_shard_size)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(output))
    print(f"Merged model written to: {output}", flush=True)


if __name__ == "__main__":
    main()
