"""Deterministic task selection shared by evaluation entry points."""

import random
from typing import Any, Dict, List


SEQUENTIAL_MODE = "sequential"
RANDOM_MODE = "random"
SUPPORTED_MODES = {SEQUENTIAL_MODE, RANDOM_MODE}


def select_task_ids(agent_config: Dict[str, Any]) -> List[int]:
    """Select task IDs from ``range(task_nums)`` using config-defined policy.

    Backward compatibility: without ``task_selection``, all task IDs are
    returned in ascending order, matching the original evaluator behavior.
    """
    task_pool_size = _non_negative_int(
        agent_config.get("task_nums", 0), "agent_config.task_nums"
    )
    selection = agent_config.get("task_selection") or {}
    if not isinstance(selection, dict):
        raise ValueError("agent_config.task_selection must be a mapping")

    mode = str(selection.get("mode", SEQUENTIAL_MODE)).strip().lower()
    if mode not in SUPPORTED_MODES:
        choices = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"task_selection.mode must be one of: {choices}")

    sample_size = _non_negative_int(
        selection.get("sample_size", task_pool_size),
        "agent_config.task_selection.sample_size",
    )
    if sample_size > task_pool_size:
        raise ValueError(
            "task_selection.sample_size cannot exceed agent_config.task_nums"
        )

    candidates = list(range(task_pool_size))
    if mode == SEQUENTIAL_MODE:
        return candidates[:sample_size]

    seed = selection.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("agent_config.task_selection.seed must be an integer")
    return random.Random(seed).sample(candidates, sample_size)


def task_selection_metadata(agent_config: Dict[str, Any], task_ids: List[int]) -> Dict[str, Any]:
    """Build a serializable record of the effective task selection."""
    selection = agent_config.get("task_selection") or {}
    mode = str(selection.get("mode", SEQUENTIAL_MODE)).strip().lower()
    return {
        "mode": mode,
        "seed": selection.get("seed", 0) if mode == RANDOM_MODE else None,
        "task_pool_size": agent_config.get("task_nums", 0),
        "sample_size": len(task_ids),
        "selected_task_ids": task_ids,
    }


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value
