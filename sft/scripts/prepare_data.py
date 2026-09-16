#!/usr/bin/env python
"""Remove the Thought instruction block from ShopSimulator SFT records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


THOUGHT_INSTRUCTION = """请在 Thought 中尝试回答以下几个问题：
在当前的状态下你是如何思考以做出下一步操作？
应该选择执行什么操作？
这个操作是否直接或间接地帮助实现最终目标？
选择的这个商品是否能够完全符合用户的要求？
选择的商品规格是否完全覆盖了用户的要求？
请你认真思考后给出你的思考过程。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def clean_record(record: dict[str, Any], line_number: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"line {line_number}: messages must be a list")

    system_messages = [message for message in messages if message.get("role") == "system"]
    if len(system_messages) != 1:
        raise ValueError(
            f"line {line_number}: expected one system message, got {len(system_messages)}"
        )

    system_message = system_messages[0]
    content = system_message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"line {line_number}: system content must be a string")
    if content.count(THOUGHT_INSTRUCTION) != 1:
        raise ValueError(
            f"line {line_number}: expected the Thought instruction exactly once"
        )

    cleaned = content.replace(THOUGHT_INSTRUCTION, "")
    system_message["content"] = cleaned.rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with input_path.open(encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for count, line in enumerate(source, 1):
                record = json.loads(line)
                clean_record(record, count)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Wrote {count} cleaned records to {output_path}")


if __name__ == "__main__":
    main()
