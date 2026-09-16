"""Convert verified trajectories to action-only Qwen3.5 tool-calling data."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tool_adapter import SHOP_TOOLS
from trajectory_collection.verifier import parse_arguments


PRIVILEGED_TOOL_MARKERS = ("Reward Details [SEP]", "Target [SEP] asin")


def export_messages(
    trajectory: Mapping[str, Any], *, include_reasoning: bool = False
) -> List[Dict[str, Any]]:
    exported: List[Dict[str, Any]] = []
    conversation = trajectory["conversation"]
    for index, source_message in enumerate(conversation):
        role = source_message["role"]
        if role == "assistant":
            tool_calls = source_message.get("tool_calls") or []
            if len(tool_calls) != 1:
                raise ValueError("Verified assistant message must contain one tool call")
            source_call = tool_calls[0]
            function = source_call["function"]
            assistant = {
                "role": "assistant",
                "content": str(source_message.get("content") or "")
                if include_reasoning
                else "",
                "tool_calls": [
                    {
                        "id": source_call["id"],
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": parse_arguments(source_call),
                        },
                    }
                ],
            }
            if include_reasoning and source_message.get("reasoning_content") is not None:
                assistant["reasoning_content"] = source_message["reasoning_content"]
            exported.append(assistant)
        elif role == "tool":
            # The terminal environment response contains reward and target fields.
            # Keep the tool-call pairing but never expose that privileged payload
            # to the student dataset.
            content = "" if index == len(conversation) - 1 else str(
                source_message.get("content") or ""
            )
            exported.append(
                {
                    "role": "tool",
                    "tool_call_id": source_message["tool_call_id"],
                    "content": content,
                }
            )
        elif role in {"system", "user"}:
            exported.append(
                {"role": role, "content": str(source_message.get("content") or "")}
            )
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return exported


def export_sample(trajectory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "messages": export_messages(trajectory),
        "tools": copy.deepcopy(SHOP_TOOLS),
    }


def export_reasoning_sample(trajectory: Mapping[str, Any]) -> Dict[str, Any]:
    """Export the same curated trajectory while retaining teacher reasoning."""
    return {
        "messages": export_messages(trajectory, include_reasoning=True),
        "tools": copy.deepcopy(SHOP_TOOLS),
    }


def validate_exported_sample(
    sample: Mapping[str, Any], *, allow_reasoning: bool = False
) -> None:
    if set(sample) != {"messages", "tools"}:
        raise ValueError("Training sample must contain only messages and tools")
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        role = message.get("role")
        if role == "tool" and any(
            marker in str(message.get("content") or "")
            for marker in PRIVILEGED_TOOL_MARKERS
        ):
            raise ValueError("Privileged terminal environment data leaked into SFT sample")
        if role == "assistant":
            if not allow_reasoning and message.get("content") != "":
                raise ValueError("Assistant content must be empty for action-only SFT")
            if not allow_reasoning and "reasoning_content" in message:
                raise ValueError("reasoning_content leaked into SFT sample")
            if allow_reasoning and "reasoning_content" in message and not isinstance(
                message["reasoning_content"], str
            ):
                raise ValueError("reasoning_content must be a string")
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                raise ValueError("Assistant must contain exactly one tool call")
            arguments = calls[0].get("function", {}).get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("Qwen tool arguments must be an object")
        serialized = json.dumps(message, ensure_ascii=False)
        if not allow_reasoning and (
            "<think>" in serialized or "reasoning_content" in serialized
        ):
            raise ValueError("Thinking content leaked into SFT sample")


def validate_jsonl(
    path: Path, tokenizer: Any = None, *, allow_reasoning: bool = False
) -> Dict[str, Any]:
    """Validate every row and optionally render it with a native chat template."""
    rows = 0
    assistant_actions = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                validate_exported_sample(sample, allow_reasoning=allow_reasoning)
                if tokenizer is not None:
                    rendered = tokenizer.apply_chat_template(
                        sample["messages"],
                        tools=sample["tools"],
                        tokenize=False,
                        enable_thinking=False,
                    )
                    if not isinstance(rendered, str) or not rendered:
                        raise ValueError("Native chat template rendered empty output")
            except Exception as error:
                raise ValueError(f"Invalid SFT row {line_number}: {error}") from error
            rows += 1
            assistant_actions += sum(
                message.get("role") == "assistant" for message in sample["messages"]
            )
    if rows == 0:
        raise ValueError("SFT JSONL is empty")
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "assistant_actions": assistant_actions,
        "native_chat_template_checked": tokenizer is not None,
        "enable_thinking": False,
        "reasoning_retained": allow_reasoning,
    }
