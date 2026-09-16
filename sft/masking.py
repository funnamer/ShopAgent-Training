"""Chat-template-aware assistant-only labels for tool-use SFT."""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def build_non_thinking_template(tokenizer: Any) -> str:
    """Make Qwen retain its empty non-thinking prefix on assistant history."""
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("Tokenizer has no chat_template")

    plain_assistant = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    if template.count(plain_assistant) != 2:
        raise ValueError(
            "Unsupported chat template: expected two plain assistant render "
            "expressions. Refusing to guess non-thinking prefix placement."
        )
    non_thinking_assistant = """{{- '<|im_start|>' + message.role + '\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
            {{- content }}"""
    template = template.replace(plain_assistant, non_thinking_assistant)
    return (
        "{%- set enable_thinking = enable_thinking | default(false) %}\n"
        + template
    )


def build_assistant_only_template(tokenizer: Any) -> str:
    """Add assistant-mask spans to the explicit non-thinking template.

    The shipped Qwen3 template serializes tool calls correctly but does not mark
    assistant spans, so return_assistant_tokens_mask=True produces an all-zero
    mask. It also inserts the empty non-thinking prefix only for a live
    ``add_generation_prompt``; historical assistant tool calls do not retain the
    prefix. We instrument the in-memory template so every supervised action has
    the same prefix that the model receives immediately before generation at
    inference time. Tokenizer/model files remain unchanged.
    """
    template = build_non_thinking_template(tokenizer)

    assistant_branch = '{%- elif message.role == "assistant" %}'
    tool_branch = '{%- elif message.role == "tool" %}'
    if template.count(assistant_branch) != 1 or template.count(tool_branch) != 1:
        raise ValueError(
            "Unsupported chat template: expected exactly one assistant branch "
            "followed by one tool branch. Refusing to guess token boundaries."
        )

    return template.replace(
        assistant_branch,
        assistant_branch + "\n        {%- generation %}",
    ).replace(
        tool_branch,
        "        {%- endgeneration %}\n    " + tool_branch,
    )


def validate_action_only(messages: list[dict[str, Any]]) -> None:
    """Reject data that is not truly action-only before tokenization."""
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("reasoning_content"):
            raise ValueError(
                f"Assistant message {index} contains reasoning_content"
            )
        if message.get("content"):
            raise ValueError(
                f"Assistant message {index} contains text instead of only actions"
            )
        if not message.get("tool_calls"):
            raise ValueError(
                f"Assistant message {index} has no tool_calls"
            )


def mask_supplied_assistant_prefixes(
    input_ids: list[int],
    assistant_mask: list[int],
    tokenizer: Any,
    expected_count: int,
) -> int:
    """Mask tokens supplied by the non-thinking inference chat template.

    The end-of-turn token remains trainable because learning when to stop is part
    of the target. The assistant header and empty think block are prompt tokens,
    so neither is a prediction target.
    """
    prefix_ids = tokenizer.encode(
        "<|im_start|>assistant\n" + NON_THINKING_PREFIX,
        add_special_tokens=False,
    )
    if not prefix_ids:
        raise RuntimeError(
            "Assistant non-thinking prefix tokenized to an empty sequence"
        )

    count = 0
    width = len(prefix_ids)
    for start in range(len(input_ids) - width + 1):
        if (
            input_ids[start : start + width] == prefix_ids
            and all(assistant_mask[start : start + width])
        ):
            assistant_mask[start : start + width] = [0] * width
            count += 1
    if count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} trainable assistant non-thinking "
            f"prefixes, found {count}; refusing a misaligned loss mask"
        )
    return count


def mask_truncated_action(
    input_ids: list[int],
    assistant_mask: list[int],
    tokenizer: Any,
) -> bool:
    """Drop a final assistant target if truncation cut it before end-of-turn."""
    if not assistant_mask or not assistant_mask[-1]:
        return False

    eos_token_id = tokenizer.eos_token_id
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    target_complete = input_ids[-1] == eos_token_id
    if newline_ids and len(input_ids) > len(newline_ids):
        target_complete = target_complete or (
            input_ids[-len(newline_ids) :] == newline_ids
            and input_ids[-len(newline_ids) - 1] == eos_token_id
        )
    if target_complete:
        return False

    start = len(assistant_mask) - 1
    while start > 0 and assistant_mask[start - 1]:
        start -= 1
    assistant_mask[start:] = [0] * (len(assistant_mask) - start)
    return True


def encode_assistant_only(
    record: dict[str, Any],
    tokenizer: Any,
    chat_template: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one conversation and label only assistant action payloads/EOT."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each row must contain a non-empty messages list")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("Conversation has no assistant message")
    validate_action_only(messages)

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=record.get("tools"),
        chat_template=chat_template,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=False,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_mask")
    if assistant_mask is None:
        raise RuntimeError("Tokenizer did not return an assistant mask")

    assistant_mask = list(assistant_mask)
    if not (len(input_ids) == len(attention_mask) == len(assistant_mask)):
        raise RuntimeError(
            "input_ids, attention_mask, and assistant mask differ in length"
        )
    if not input_ids:
        raise ValueError("Tokenized conversation is empty")
    if any(value not in (0, 1, False, True) for value in assistant_mask):
        raise RuntimeError("Assistant mask contains values other than 0/1")

    assistant_count = sum(
        message.get("role") == "assistant" for message in messages
    )
    mask_supplied_assistant_prefixes(
        input_ids,
        assistant_mask,
        tokenizer,
        assistant_count,
    )

    # Truncate only after masking complete prefixes. Truncating in the tokenizer
    # can cut a prefix in half and accidentally train on assistant/think tokens.
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    assistant_mask = assistant_mask[:max_length]
    mask_truncated_action(input_ids, assistant_mask, tokenizer)
    if not any(assistant_mask):
        raise ValueError(
            "No assistant action tokens remain after templating/truncation; "
            "increase max_length or inspect this sample"
        )

    labels = [
        token if is_assistant else IGNORE_INDEX
        for token, is_assistant in zip(input_ids, assistant_mask)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def masking_stats(example: dict[str, list[int]]) -> dict[str, int | float]:
    total = len(example["labels"])
    trained = sum(label != IGNORE_INDEX for label in example["labels"])
    return {
        "tokens": total,
        "trained_tokens": trained,
        "masked_tokens": total - trained,
        "trained_fraction": trained / total,
    }
