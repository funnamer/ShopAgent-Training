#!/usr/bin/env python
"""Integration checks against the real Qwen tokenizer and action-only data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from masking import (
    IGNORE_INDEX,
    NON_THINKING_PREFIX,
    build_assistant_only_template,
    build_non_thinking_template,
    encode_assistant_only,
)

MODEL = "/root/autodl-tmp/Qwen3-4B"
DATA = str(
    Path(__file__).resolve().parents[2]
    / "shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft_no_thought_prompt.jsonl"
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=False,
        use_fast=True,
    )
    template = build_assistant_only_template(tokenizer)
    inference_template = build_non_thinking_template(tokenizer)
    checked = 0

    with open(DATA, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                record["messages"],
                tools=record["tools"],
                chat_template=template,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            assistant_count = sum(
                message.get("role") == "assistant"
                for message in record["messages"]
            )
            expected_prefix = "<|im_start|>assistant\n" + NON_THINKING_PREFIX
            assert rendered.count(expected_prefix) == assistant_count

            first_assistant = next(
                index
                for index, message in enumerate(record["messages"])
                if message.get("role") == "assistant"
            )
            inference_prompt = tokenizer.apply_chat_template(
                record["messages"][:first_assistant],
                tools=record["tools"],
                chat_template=inference_template,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert inference_prompt.endswith(expected_prefix)
            default_inference_prompt = tokenizer.apply_chat_template(
                record["messages"][:first_assistant],
                tools=record["tools"],
                chat_template=inference_template,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert default_inference_prompt == inference_prompt

            inference_history = tokenizer.apply_chat_template(
                record["messages"],
                tools=record["tools"],
                chat_template=inference_template,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            assert inference_history.count(expected_prefix) == assistant_count

            encoded = encode_assistant_only(
                record,
                tokenizer,
                template,
                max_length=8192,
            )
            labels = encoded["labels"]
            assert len(encoded["input_ids"]) == len(labels)
            assert any(label != IGNORE_INDEX for label in labels)
            assert any(label == IGNORE_INDEX for label in labels)

            learned_ids = [
                token
                for token, label in zip(encoded["input_ids"], labels)
                if label != IGNORE_INDEX
            ]
            learned_text = tokenizer.decode(
                learned_ids,
                skip_special_tokens=False,
            )
            assert "<|im_start|>assistant" not in learned_text
            assert "<think>" not in learned_text
            assert "</think>" not in learned_text
            assert "<tool_call>" in learned_text
            assert "<tool_response>" not in learned_text
            assert "<|im_start|>system" not in learned_text
            assert "<|im_start|>user" not in learned_text

            for message in record["messages"]:
                if message.get("role") == "tool" and message.get("content"):
                    probe = message["content"][:40]
                    assert probe not in learned_text

            checked += 1
            if checked == 32:
                break

    assert checked == 32

    # A cut inside an action must not teach an incomplete tool-call JSON target.
    with open(DATA, encoding="utf-8") as handle:
        long_record = next(
            json.loads(line) for index, line in enumerate(handle) if index == 213
        )
    truncated = encode_assistant_only(
        long_record,
        tokenizer,
        template,
        max_length=8192,
    )
    assert truncated["labels"][-1] == IGNORE_INDEX
    print(f"Masking integration test passed for {checked} real samples.")


if __name__ == "__main__":
    main()
