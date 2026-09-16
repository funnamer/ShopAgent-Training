#!/usr/bin/env python3
"""Estimate trajectory context and reasoning lengths using character counts.

The context estimate is the compact-JSON character count of the tool schema plus
all messages preceding each assistant turn.  The largest prefix is reported for
each trajectory.  This deliberately avoids tokenizer/model dependencies and is
intended for coarse filtering and visualization.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECTION_ROOT = SCRIPT_DIR.parent / "outputs" / "standard-single-deepseek-thinking-test50-v2"
DEFAULT_INPUT = COLLECTION_ROOT / "sft" / "accepted_reasoning_sft.jsonl"
DEFAULT_MANIFEST = COLLECTION_ROOT / "sft" / "selected_manifest_candidate_ids.jsonl"


def compact_char_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def percentile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def analyze_row(
    row: dict[str, Any], row_index: int, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    messages = row.get("messages")
    tools = row.get("tools", [])
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index} has no messages")

    context_lengths = []
    reasoning_lengths = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        # Approximate the prompt visible immediately before this assistant turn.
        context_lengths.append(
            compact_char_count({"tools": tools, "messages": messages[:message_index]})
        )
        reasoning = message.get("reasoning_content") or ""
        reasoning_lengths.append(len(str(reasoning)))

    if not context_lengths:
        raise ValueError(f"row {row_index} has no assistant turns")

    result = {
        "row_index": row_index,
        "task_id": "" if metadata is None else metadata.get("task_id", ""),
        "rollout_id": "" if metadata is None else metadata.get("rollout_id", ""),
        "difficulty": "" if metadata is None else metadata.get("difficulty", ""),
        "assistant_turns": len(context_lengths),
        "max_context_chars": max(context_lengths),
        "trajectory_total_chars": compact_char_count(row),
        "max_single_reasoning_chars": max(reasoning_lengths, default=0),
        "total_reasoning_chars": sum(reasoning_lengths),
    }
    return result


def plot_stats(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required only for plotting; run this script in an "
            "environment that provides matplotlib"
        ) from error

    contexts = [row["max_context_chars"] for row in rows]
    reasoning = [row["max_single_reasoning_chars"] for row in rows]
    turns = [row["assistant_turns"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Trajectory Length Analysis ({len(rows):,} samples, character estimate)")

    axes[0, 0].hist(
        contexts, bins=80, range=(0, 40000), color="#3b82f6", edgecolor="white"
    )
    axes[0, 0].axvline(percentile(contexts, 0.95), color="#dc2626", linestyle="--", label="P95")
    axes[0, 0].set_title("Maximum context per trajectory")
    axes[0, 0].set_xlabel("Characters")
    axes[0, 0].set_ylabel("Trajectories")
    axes[0, 0].set_xlim(0, 40000)
    axes[0, 0].legend()

    axes[0, 1].hist(
        reasoning, bins=80, range=(0, 10000), color="#8b5cf6", edgecolor="white"
    )
    axes[0, 1].axvline(percentile(reasoning, 0.95), color="#dc2626", linestyle="--", label="P95")
    axes[0, 1].set_title("Longest single reasoning segment")
    axes[0, 1].set_xlabel("Characters")
    axes[0, 1].set_ylabel("Trajectories")
    axes[0, 1].set_xlim(0, 10000)
    axes[0, 1].legend()

    axes[1, 0].scatter(contexts, reasoning, c=turns, cmap="viridis", alpha=0.55, s=18)
    axes[1, 0].set_title("Context vs. longest reasoning")
    axes[1, 0].set_xlabel("Maximum context characters")
    axes[1, 0].set_ylabel("Maximum reasoning characters")

    sorted_contexts = sorted(contexts)
    cumulative = [(index + 1) / len(sorted_contexts) * 100 for index in range(len(sorted_contexts))]
    axes[1, 1].plot(sorted_contexts, cumulative, color="#059669", linewidth=2)
    axes[1, 1].axhline(95, color="#dc2626", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Context-length cumulative distribution")
    axes[1, 1].set_xlabel("Maximum context characters")
    axes[1, 1].set_ylabel("Trajectories covered (%)")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_steps(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required only for plotting; run this script in an "
            "environment that provides matplotlib"
        ) from error

    counts = Counter(row["assistant_turns"] for row in rows)
    steps = list(range(min(counts), max(counts) + 1))
    frequencies = [counts.get(step, 0) for step in steps]
    total = len(rows)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(steps, frequencies, color="#0ea5e9", edgecolor="white", width=0.82)
    ax.set_title(f"Steps per Trajectory ({total:,} samples)")
    ax.set_xlabel("Steps (assistant tool calls)")
    ax.set_ylabel("Trajectories")
    ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.25)

    for bar, frequency in zip(bars, frequencies):
        if frequency == 0:
            continue
        ax.annotate(
            f"{frequency}\n{frequency / total:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max(frequencies) * 1.16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "trajectory_length_distribution.png",
    )
    parser.add_argument(
        "--steps-output",
        type=Path,
        default=SCRIPT_DIR / "trajectory_steps_distribution.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    metadata_rows = load_jsonl(args.manifest) if args.manifest.exists() else []
    if metadata_rows and len(metadata_rows) != len(samples):
        raise ValueError(
            f"manifest has {len(metadata_rows)} rows but input has {len(samples)} rows"
        )

    analyzed = [
        analyze_row(sample, index, metadata_rows[index - 1] if metadata_rows else None)
        for index, sample in enumerate(samples, 1)
    ]
    context_values = [row["max_context_chars"] for row in analyzed]
    reasoning_values = [row["max_single_reasoning_chars"] for row in analyzed]
    total_values = [row["trajectory_total_chars"] for row in analyzed]
    summary = {
        "input": str(args.input.resolve()),
        "method": (
            "Character estimate. Context is compact JSON for tools plus all messages "
            "preceding each assistant turn; max is taken within each trajectory."
        ),
        "trajectory_count": len(analyzed),
        "steps": describe([row["assistant_turns"] for row in analyzed]),
        "max_context_chars": describe(context_values),
        "trajectory_total_chars": describe(total_values),
        "max_single_reasoning_chars": describe(reasoning_values),
        "context_threshold_counts": {
            str(limit): sum(value > limit for value in context_values)
            for limit in (8000, 16000, 32000, 64000, 128000)
        },
        "reasoning_threshold_counts": {
            str(limit): sum(value > limit for value in reasoning_values)
            for limit in (1000, 2000, 4000, 8000, 16000)
        },
    }

    plot_stats(analyzed, args.output)
    plot_steps(analyzed, args.steps_output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Chart: {args.output.resolve()}")
    print(f"Steps chart: {args.steps_output.resolve()}")


if __name__ == "__main__":
    main()
