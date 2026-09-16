#!/usr/bin/env python
"""LoRA SFT for Qwen3 tool-use trajectories with assistant-only loss."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from masking import (
    IGNORE_INDEX,
    build_assistant_only_template,
    build_non_thinking_template,
    encode_assistant_only,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "model_path": "/root/autodl-tmp/Qwen3-4B",
    "data_path": "shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft_no_thought_prompt.jsonl",
    "output_dir": "/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora",
    "max_length": 8192,
    "eval_ratio": 0.10,
    "seed": 42,
    "num_train_epochs": 3.0,
    "max_steps": -1,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps": 5,
    "eval_steps": 25,
    "save_steps": 50,
    "save_total_limit": None,
    "dataloader_num_workers": 2,
    "swanlab_mode": "online",
    "swanlab_project": "qwen3-shop-action-sft",
    "swanlab_run_name": "qwen3-4b-action-only-lora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON configuration path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data/masks without loading the model",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test subset",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config = dict(DEFAULTS)
    with open(path, encoding="utf-8") as handle:
        supplied = json.load(handle)
    unknown = sorted(set(supplied) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    config.update(supplied)
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        config["data_path"] = str(PROJECT_ROOT / data_path)
    if config["max_length"] <= 0:
        raise ValueError("max_length must be positive")
    if not 0 < config["eval_ratio"] < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    if config["swanlab_mode"] not in {"online", "offline", "local", "disabled"}:
        raise ValueError(
            "swanlab_mode must be online, offline, local, or disabled"
        )
    for key in ("swanlab_project", "swanlab_run_name"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    return config


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        width = batch["input_ids"].shape[1]
        padded_labels = []
        for feature in features:
            labels = list(feature["labels"])
            padded_labels.append(labels + [IGNORE_INDEX] * (width - len(labels)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def quantile(sorted_values: list[int], probability: float) -> int:
    if not sorted_values:
        return 0
    index = min(
        len(sorted_values) - 1,
        math.ceil(probability * len(sorted_values)) - 1,
    )
    return sorted_values[index]


def print_dataset_stats(dataset: Dataset, max_length: int) -> None:
    lengths = sorted(len(row) for row in dataset["input_ids"])
    trained = [
        sum(label != IGNORE_INDEX for label in row)
        for row in dataset["labels"]
    ]
    print(
        json.dumps(
            {
                "samples": len(dataset),
                "tokens_p50": quantile(lengths, 0.50),
                "tokens_p90": quantile(lengths, 0.90),
                "tokens_p95": quantile(lengths, 0.95),
                "tokens_max_after_truncation": max(lengths),
                "samples_at_max_length": sum(
                    length == max_length for length in lengths
                ),
                "trained_tokens_total": sum(trained),
                "trained_tokens_per_sample_mean": round(
                    sum(trained) / len(trained), 2
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def prepare_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    max_samples: int | None,
) -> Dataset:
    data_path = Path(config["data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    raw = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        raw = raw.select(range(min(max_samples, len(raw))))

    chat_template = build_assistant_only_template(tokenizer)

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        return encode_assistant_only(
            row,
            tokenizer,
            chat_template,
            config["max_length"],
        )

    tokenized = raw.map(
        tokenize,
        remove_columns=raw.column_names,
        desc="Applying Qwen chat template and assistant-only mask",
        # The mapper calls masking helpers from another module. Datasets may not
        # invalidate its Arrow cache when those helper implementations change.
        load_from_cache_file=False,
    )
    print_dataset_stats(tokenized, config["max_length"])
    return tokenized


def validate_lora_targets(model: torch.nn.Module, targets: list[str]) -> None:
    module_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
    }
    missing = sorted(set(targets) - module_suffixes)
    if missing:
        raise ValueError(f"LoRA target modules not found in model: {missing}")


def initialize_swanlab(
    config: dict[str, Any],
    output_dir: Path,
    train_samples: int,
    eval_samples: int,
) -> Any:
    """Start SwanLab before Transformers' callback emits its first event."""
    try:
        import swanlab
    except ImportError as error:
        raise RuntimeError(
            "SwanLab is required for training; install requirements in the ft environment"
        ) from error

    run_config = dict(config)
    run_config.update(
        {
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "effective_batch_size": (
                config["per_device_train_batch_size"]
                * config["gradient_accumulation_steps"]
            ),
        }
    )
    return swanlab.init(
        mode=config["swanlab_mode"],
        project=config["swanlab_project"],
        name=config["swanlab_run_name"],
        log_dir=str(output_dir / "swanlog"),
        config=run_config,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    dataset = prepare_dataset(config, tokenizer, args.max_samples)
    # Persist the inference form of the same template in checkpoints/final output.
    # It has no Transformers-only generation-mask tags.
    tokenizer.chat_template = build_non_thinking_template(tokenizer)
    if args.validate_only:
        print(
            "Validation passed: every retained sample has "
            "non-empty assistant-only labels."
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    split = dataset.train_test_split(
        test_size=config["eval_ratio"],
        seed=config["seed"],
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    validate_lora_targets(model, config["lora_target_modules"])
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["lora_target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "chat_template_non_thinking.jinja",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(tokenizer.chat_template)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    initialize_swanlab(
        config,
        output_dir,
        train_samples=len(train_dataset),
        eval_samples=len(eval_dataset),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type="cosine",
        # Transformers 5.x accepts a float in [0, 1) here as a ratio.
        warmup_steps=config["warmup_ratio"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["swanlab"],
        run_name=config["swanlab_run_name"],
        seed=config["seed"],
        data_seed=config["seed"],
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        length_column_name="length",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    try:
        result = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        trainer.save_model(str(output_dir / "final_adapter"))
        tokenizer.save_pretrained(str(output_dir / "final_adapter"))
        trainer.save_state()
        print(json.dumps(result.metrics, indent=2))
    finally:
        import swanlab

        swanlab.finish()


if __name__ == "__main__":
    main()
