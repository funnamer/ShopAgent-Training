#!/usr/bin/env python
"""Decode labeled and masked regions for a human-readable masking audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from masking import (
    IGNORE_INDEX,
    build_assistant_only_template,
    encode_assistant_only,
    masking_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft_no_thought_prompt.jsonl"),
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-chars", type=int, default=5000)
    args = parser.parse_args()

    if args.row < 0:
        raise ValueError("--row must be non-negative")
    with open(args.data, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == args.row:
                record = json.loads(line)
                break
        else:
            raise IndexError(f"Row {args.row} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
        use_fast=True,
    )
    template = build_assistant_only_template(tokenizer)
    encoded = encode_assistant_only(
        record,
        tokenizer,
        template,
        args.max_length,
    )
    learned_ids = [
        token
        for token, label in zip(encoded["input_ids"], encoded["labels"])
        if label != IGNORE_INDEX
    ]
    masked_ids = [
        token
        for token, label in zip(encoded["input_ids"], encoded["labels"])
        if label == IGNORE_INDEX
    ]

    print(json.dumps(masking_stats(encoded), indent=2))
    print("\n=== LEARNED TOKENS (assistant actions/text only) ===")
    print(tokenizer.decode(learned_ids, skip_special_tokens=False)[: args.max_chars])
    print("\n=== MASKED TOKENS (context only; concatenated for audit) ===")
    print(tokenizer.decode(masked_ids, skip_special_tokens=False)[: args.max_chars])


if __name__ == "__main__":
    main()
