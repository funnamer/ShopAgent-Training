"""Add ShopSimulator reward diagnostics to verl's scalar logger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


_PATCH_MARKER = "_shopsimulator_reward_metrics"


def _numeric_summary(metrics: dict[str, float], prefix: str, values: Any) -> None:
    array = np.asarray(values)
    try:
        array = array.astype(np.float64)
    except (TypeError, ValueError):
        return
    array = array[np.isfinite(array)]
    if array.size:
        metrics[f"{prefix}/mean"] = float(np.mean(array))
        metrics[f"{prefix}/std"] = float(np.std(array))
        metrics[f"{prefix}/min"] = float(np.min(array))
        metrics[f"{prefix}/max"] = float(np.max(array))


def _categorical_rates(metrics: dict[str, float], prefix: str, values: Any) -> None:
    array = np.asarray(values, dtype=object)
    if not array.size:
        return
    counts = defaultdict(int)
    for value in array.tolist():
        counts[str(value)] += 1
    for value, count in counts.items():
        safe_value = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
        metrics[f"{prefix}/{safe_value}_rate"] = count / float(array.size)


def add_reward_metrics(metrics: dict[str, Any], batch) -> dict[str, Any]:
    """Aggregate per-sample reward fields into step-level scalar metrics."""
    non_tensor = batch.non_tensor_batch
    numeric_fields = (
        "score", "strict_success", "type_ok", "attr_score", "option_score",
        "price_ok", "purchase_quality", "best_seen_quality", "regret",
        "repeat_count", "invalid_count", "back_count", "action_count",
    )
    for field in numeric_fields:
        if field in non_tensor:
            _numeric_summary(metrics, f"reward/{field}", non_tensor[field])

    if "strict_success" in non_tensor:
        values = np.asarray(non_tensor["strict_success"], dtype=np.float64)
        metrics["reward/success_rate"] = float(np.mean(values)) if values.size else 0.0
    if "outcome" in non_tensor:
        _categorical_rates(metrics, "reward/outcome", non_tensor["outcome"])
    if "termination" in non_tensor:
        _categorical_rates(metrics, "reward/termination", non_tensor["termination"])

    if "uid" in non_tensor and "score" in non_tensor:
        uids = np.asarray(non_tensor["uid"], dtype=object)
        scores = np.asarray(non_tensor["score"], dtype=np.float64)
        grouped: dict[str, list[float]] = defaultdict(list)
        for uid, score in zip(uids.tolist(), scores.tolist(), strict=False):
            grouped[str(uid)].append(float(score))
        stds = [float(np.std(values)) for values in grouped.values() if len(values) > 1]
        if stds:
            metrics["reward/group_score_std_mean"] = float(np.mean(stds))
            metrics["reward/group_zero_variance_rate"] = float(np.mean(np.asarray(stds) == 0.0))
    return metrics


def install() -> None:
    """Install the wrapper in the Ray TaskRunner process via runtime_env."""
    import verl.trainer.ppo.ray_trainer as ray_trainer

    if getattr(ray_trainer, _PATCH_MARKER, False):
        return
    original = ray_trainer.compute_data_metrics

    def compute_data_metrics_with_shop_rewards(batch, use_critic=True):
        metrics = original(batch=batch, use_critic=use_critic)
        return add_reward_metrics(metrics, batch)

    ray_trainer.compute_data_metrics = compute_data_metrics_with_shop_rewards
    setattr(ray_trainer, _PATCH_MARKER, True)
