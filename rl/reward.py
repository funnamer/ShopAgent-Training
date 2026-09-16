"""Deterministic trajectory reward for ShopSimulator GRPO.

The reward is intentionally terminal-first.  It rewards a fully compliant
purchase, strongly penalizes irreversible wrong purchases, and adds only small
penalties for repeated/invalid actions and for ignoring a better item that was
actually visible during the trajectory.  It does not reward ``back`` directly
and it has no generic step penalty.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

try:
    from rapidfuzz.fuzz import partial_ratio
except ImportError:  # pragma: no cover - deterministic fallback for light tests
    from difflib import SequenceMatcher

    def partial_ratio(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left, right).ratio()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT / "shopSimulator/shop_env/data/fine_items_eval_train_all.json"
)


class InfrastructureRewardError(RuntimeError):
    """Signals a rollout that must be discarded instead of training on it."""


class RewardConfig:
    _defaults = {
        "attr_weight": 0.45,
        "option_weight": 0.40,
        "price_weight": 0.15,
        "incorrect_quality_scale": 0.20,
        "regret_scale": 0.20,
        "repeat_penalty": 0.05,
        "repeat_penalty_cap": 0.20,
        "invalid_penalty": 0.05,
        "invalid_penalty_cap": 0.10,
        "correct_purchase_reward": 1.0,
        "incorrect_purchase_reward": -1.0,
        "timeout_reward": -1.05,
        "invalid_model_reward": -1.10,
        "min_reward": -1.50,
        "max_reward": 1.00,
        "fuzzy_threshold": 85.0,
    }

    def __init__(self, **overrides: float) -> None:
        unknown = set(overrides) - set(self._defaults)
        if unknown:
            raise TypeError(f"unknown reward settings: {sorted(unknown)}")
        for name, value in self._defaults.items():
            setattr(self, name, float(overrides.get(name, value)))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self._defaults}


class Compliance(NamedTuple):
    type_ok: float
    attr_score: float
    option_score: float
    price_ok: float
    quality: float
    strict_success: bool


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _category(value: Any) -> tuple[str, ...]:
    return tuple(_norm(part) for part in re.split(r"[›>/]+", str(value or "")) if _norm(part))


def strict_type_match(product: Mapping[str, Any], goal: Mapping[str, Any]) -> bool:
    """Hard type gate: exact ASIN or exact non-empty category path.

    Empty queries are never considered evidence, and broad category overlap is
    deliberately insufficient for an irreversible purchase.
    """

    product_asin = _norm(product.get("asin"))
    goal_asin = _norm(goal.get("asin"))
    if product_asin and goal_asin and product_asin == goal_asin:
        return True
    product_category = _category(product.get("category"))
    goal_category = _category(goal.get("category"))
    return bool(product_category and goal_category and product_category == goal_category)


def _numbers(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+(?:\.\d+)?", value))


def _text_match(wanted: str, candidate: str, threshold: float) -> bool:
    # Fuzzy text matching must never turn 50W into 500W, 12V into 24V, etc.
    wanted_numbers = _numbers(wanted)
    candidate_numbers = _numbers(candidate)
    if wanted_numbers and wanted_numbers != candidate_numbers:
        return False
    return wanted in candidate or partial_ratio(wanted, candidate) >= threshold


def _match_fraction(required: Sequence[Any], available: Iterable[Any], threshold: float) -> float:
    required_norm = [_norm(item) for item in required if _norm(item)]
    if not required_norm:
        return 1.0
    available_norm = [_norm(item) for item in available if _norm(item)]
    corpus = "".join(available_norm)
    matched = 0
    for wanted in required_norm:
        # Corpus substring is useful for non-numeric attributes split across
        # fields; numeric constraints are checked against individual fields.
        if (not _numbers(wanted) and wanted in corpus) or any(
            candidate and _text_match(wanted, candidate, threshold) for candidate in available_norm
        ):
            matched += 1
    return matched / len(required_norm)


def _product_text(product: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = [
        product.get("name"),
        product.get("title"),
        product.get("Title"),
        product.get("full_description"),
        product.get("Description"),
    ]
    for key in ("attributes", "attribute", "Attributes"):
        item = product.get(key)
        if isinstance(item, list):
            values.extend(item)
        elif item:
            values.append(item)
    return values


def _option_values(product: Mapping[str, Any]) -> list[Any]:
    result: list[Any] = []
    options = product.get("options")
    if isinstance(options, Mapping):
        for value in options.values():
            result.extend(value if isinstance(value, list) else [value])
    elif isinstance(options, list):
        result.extend(options)
    customization = product.get("customization_options")
    if isinstance(customization, Mapping):
        for values in customization.values():
            for item in values or []:
                result.append(item.get("value") if isinstance(item, Mapping) else item)
    return result


def _option_records(product: Mapping[str, Any]) -> list[tuple[Any, float | None]]:
    result: list[tuple[Any, float | None]] = []
    customization = product.get("customization_options")
    if isinstance(customization, Mapping):
        for values in customization.values():
            for item in values or []:
                if isinstance(item, Mapping):
                    price = item.get("price")
                    result.append((item.get("value"), None if price is None else float(price)))
    if not result:
        result.extend((value, None) for value in _option_values(product))
    return result


def _selected_values(selected_options: Any) -> list[Any]:
    if isinstance(selected_options, Mapping):
        return list(selected_options.values())
    if isinstance(selected_options, list):
        return selected_options
    return []


def _all_prices(product: Mapping[str, Any]) -> list[float]:
    prices: list[float] = []
    customization = product.get("customization_options")
    if isinstance(customization, Mapping):
        for values in customization.values():
            for item in values or []:
                if isinstance(item, Mapping) and item.get("price") is not None:
                    prices.append(float(item["price"]))
    for value in product.get("pricing") or []:
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            pass
    return prices


def evaluate_compliance(
    goal: Mapping[str, Any],
    product: Mapping[str, Any],
    *,
    selected_options: Any = None,
    purchase_price: Any = None,
    optimistic: bool = False,
    config: RewardConfig = RewardConfig(),
) -> Compliance:
    """Score a purchased product or the best feasible SKU of a seen product."""

    type_ok = float(strict_type_match(product, goal))
    attr_score = _match_fraction(
        list(goal.get("attributes") or []), _product_text(product), config.fuzzy_threshold
    )
    available_options = _option_values(product) if optimistic else _selected_values(selected_options)
    option_score = _match_fraction(
        list(goal.get("goal_options") or []), available_options, config.fuzzy_threshold
    )

    upper = goal.get("price_upper")
    if upper is None:
        price_ok = 1.0
    elif optimistic:
        records = _option_records(product)
        wanted_options = [_norm(x) for x in goal.get("goal_options") or [] if _norm(x)]
        matched_prices = [
            price
            for value, price in records
            if price is not None
            and any(_text_match(wanted, _norm(value), config.fuzzy_threshold) for wanted in wanted_options)
        ]
        # When a required SKU option exists, its own price must fit; the price
        # of an unrelated cheap variant is not evidence of compliance.
        prices = matched_prices if wanted_options and matched_prices else _all_prices(product)
        price_ok = float(bool(prices) and min(prices) <= float(upper) + 1e-8)
    else:
        try:
            price_ok = float(purchase_price is not None and float(purchase_price) <= float(upper) + 1e-8)
        except (TypeError, ValueError):
            price_ok = 0.0

    quality = type_ok * (
        config.attr_weight * attr_score
        + config.option_weight * option_score
        + config.price_weight * price_ok
    )
    strict_success = bool(
        type_ok == 1.0
        and math.isclose(attr_score, 1.0, abs_tol=1e-9)
        and math.isclose(option_score, 1.0, abs_tol=1e-9)
        and price_ok == 1.0
    )
    return Compliance(type_ok, attr_score, option_score, price_ok, quality, strict_success)


def _catalog_product(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asin": raw.get("asin"),
        "category": raw.get("category"),
        "title": raw.get("title"),
        "full_description": raw.get("full_description"),
        "attribute": raw.get("attribute") or [],
        "customization_options": raw.get("customization_options") or {},
        "pricing": raw.get("pricing") or [],
    }


@lru_cache(maxsize=2)
def load_catalog(path: str) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        products = json.load(handle)
    return {str(item["asin"]): _catalog_product(item) for item in products}


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def compute_trajectory_reward(
    trace: Mapping[str, Any] | None,
    ground_truth: Mapping[str, Any] | None,
    *,
    catalog_path: str | None = None,
    config: RewardConfig = RewardConfig(),
) -> dict[str, Any]:
    trace = dict(trace or {})
    ground_truth = dict(ground_truth or {})
    termination = str(trace.get("termination") or "invalid_model")
    if termination == "infrastructure_error":
        raise InfrastructureRewardError(trace.get("error") or "ShopSimulator infrastructure error")

    final_result = _as_mapping(trace.get("final_result"))
    goal = _as_mapping(final_result.get("goal")) or _as_mapping(ground_truth.get("goal")) or ground_truth
    purchase = _as_mapping(final_result.get("purchase"))

    compliance = Compliance(0.0, 0.0, 0.0, 0.0, 0.0, False)
    if purchase:
        compliance = evaluate_compliance(
            goal,
            purchase,
            selected_options=purchase.get("options"),
            purchase_price=purchase.get("price"),
            config=config,
        )

    best_seen_quality = 0.0
    seen_asins = list(dict.fromkeys(str(x) for x in trace.get("seen_asins") or []))
    if seen_asins and goal:
        selected_catalog_path = catalog_path or os.getenv("SHOPSIM_CATALOG_PATH") or str(DEFAULT_CATALOG_PATH)
        catalog = load_catalog(str(Path(selected_catalog_path).resolve()))
        for asin in seen_asins:
            candidate = catalog.get(asin)
            if candidate:
                best_seen_quality = max(
                    best_seen_quality,
                    evaluate_compliance(goal, candidate, optimistic=True, config=config).quality,
                )

    regret = max(0.0, best_seen_quality - compliance.quality) if purchase else 0.0
    repeat_count = int(trace.get("repeat_count") or 0)
    invalid_count = int(trace.get("invalid_count") or 0)
    repeat_cost = min(config.repeat_penalty_cap, config.repeat_penalty * repeat_count)
    invalid_cost = min(config.invalid_penalty_cap, config.invalid_penalty * invalid_count)
    regret_cost = config.regret_scale * regret

    if purchase:
        if compliance.strict_success:
            base_reward = config.correct_purchase_reward
            outcome = "correct_purchase"
        else:
            base_reward = config.incorrect_purchase_reward + config.incorrect_quality_scale * compliance.quality
            outcome = "incorrect_purchase"
    elif termination in {"timeout", "turn_limit", "history_limit", "context_limit"}:
        base_reward = config.timeout_reward
        outcome = "timeout"
    else:
        base_reward = config.invalid_model_reward
        outcome = "invalid_model"

    score = max(
        config.min_reward,
        min(config.max_reward, base_reward - regret_cost - repeat_cost - invalid_cost),
    )
    return {
        "score": float(score),
        "strict_success": float(compliance.strict_success),
        "outcome": outcome,
        "termination": termination,
        "type_ok": compliance.type_ok,
        "attr_score": compliance.attr_score,
        "option_score": compliance.option_score,
        "price_ok": compliance.price_ok,
        "purchase_quality": compliance.quality,
        "best_seen_quality": best_seen_quality,
        "regret": regret,
        "repeat_count": repeat_count,
        "invalid_count": invalid_count,
        "back_count": int(trace.get("back_count") or 0),
        "action_count": int(trace.get("action_count") or 0),
        "reward_config": json.dumps(config.to_dict(), sort_keys=True),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """verl custom reward entry point."""

    del data_source, solution_str
    ground_truth_dict = _as_mapping(ground_truth)
    reward_extra = dict(extra_info or {})
    trace = _as_mapping(reward_extra.get("shop_trace"))
    if not trace:
        # V1's async reward loop exposes AgentLoopOutput.extra_fields through
        # ``tool_extra_fields``. Accept that shape as well as V0's merged
        # ``extra_info`` shape.
        trace = _as_mapping(_as_mapping(reward_extra.get("tool_extra_fields")).get("shop_trace"))
    return compute_trajectory_reward(trace, ground_truth_dict)
