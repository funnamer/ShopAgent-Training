"""Executable hard checks and deterministic quality metrics for trajectories."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SEARCH_AVAILABLE_RE = re.compile(r"搜索功能是否可用:\s*(True|False)", re.IGNORECASE)
CLICKABLES_RE = re.compile(r"可点击的按钮:\s*(\[[^\n]*\])")
LEAK_MARKERS = (
    '"reward_detail"',
    '"goal_options"',
    '"price_upper"',
    "正确答案是",
    "目标商品ASIN",
)


@dataclass
class VerificationResult:
    accepted: bool
    reasons: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_value(value: Any) -> str:
    """Normalize product options while retaining every semantic character.

    The source data and rendered shop UI do not preserve punctuation exactly
    (for example, ``/`` may be rendered as ``|`` and an emoji may become the
    replacement character ``?``).  Product-option equality therefore ignores
    Unicode punctuation, symbols, and whitespace, but remains exact over all
    letters and numbers.
    """
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return "".join(character for character in text if character.isalnum())


def normalize_click_value(value: Any) -> str:
    """Match ShopEnv's case-insensitive clickable lookup semantics."""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def parse_arguments(tool_call: Mapping[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    raw = function.get("arguments")
    if isinstance(raw, dict):
        result = raw
    elif isinstance(raw, str):
        result = json.loads(raw)
    else:
        raise ValueError("tool arguments must be a JSON object or JSON string")
    if not isinstance(result, dict):
        raise ValueError("tool arguments must decode to an object")
    return result


def _state_controls(content: str) -> Tuple[Optional[bool], Optional[List[str]]]:
    availability_matches = SEARCH_AVAILABLE_RE.findall(content)
    search_available = None
    if availability_matches:
        search_available = availability_matches[-1].casefold() == "true"
    button_matches = CLICKABLES_RE.findall(content)
    buttons = None
    if button_matches:
        try:
            parsed = json.loads(button_matches[-1])
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                buttons = parsed
        except json.JSONDecodeError:
            buttons = None
    return search_available, buttons


def _goal_options(goal: Mapping[str, Any]) -> List[Any]:
    options = goal.get("goal_options") or []
    if isinstance(options, dict):
        values: List[Any] = []
        for option_values in options.values():
            if isinstance(option_values, list):
                values.extend(option_values)
            else:
                values.append(option_values)
        return values
    return list(options) if isinstance(options, list) else [options]


def _purchase_options(purchase: Mapping[str, Any]) -> List[Any]:
    options = purchase.get("options") or {}
    if isinstance(options, dict):
        return list(options.values())
    return list(options) if isinstance(options, list) else [options]


def _instruction_sha256(content: str) -> str:
    instruction = content.split("\n\n搜索功能是否可用:", 1)[0]
    if instruction.startswith("Instruction: "):
        instruction = instruction[len("Instruction: ") :]
    normalized = unicodedata.normalize("NFKC", instruction).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _summary_matches(
    source: Mapping[str, Any], summary: Mapping[str, Any], keys: Sequence[str]
) -> bool:
    return all(key not in summary or summary.get(key) == source.get(key) for key in keys)


def verify_trajectory(
    trajectory: Mapping[str, Any],
    diagnostics: Optional[Mapping[str, Any]],
    train_start: int,
    train_end: int,
    *,
    expected_task_id: Optional[int] = None,
    expected_system_prompt_sha256: Optional[str] = None,
    expected_instruction_sha256: Optional[str] = None,
    expected_model_name: Optional[str] = None,
    expected_run_name: Optional[str] = None,
) -> VerificationResult:
    reasons: List[str] = []
    task_id = trajectory.get("task_id")
    if not isinstance(task_id, int) or not train_start <= task_id < train_end:
        reasons.append("task_not_in_train_range")
    if expected_task_id is not None and task_id != expected_task_id:
        reasons.append("trajectory_task_id_mismatch")

    reward = trajectory.get("reward")
    if reward != 1 and reward != 1.0:
        reasons.append("reward_not_one")
    goal = trajectory.get("goal") or {}
    purchase = trajectory.get("purchase") or {}
    if not goal.get("asin") or purchase.get("asin") != goal.get("asin"):
        reasons.append("asin_mismatch")

    reward_detail = trajectory.get("reward_detail") or {}
    for key in ("r_att", "r_option", "r_price", "r_type"):
        if reward_detail.get(key) != 1 and reward_detail.get(key) is not True:
            reasons.append(f"{key}_not_full")

    expected_options = sorted(normalize_value(v) for v in _goal_options(goal))
    actual_options = sorted(normalize_value(v) for v in _purchase_options(purchase))
    if expected_options != actual_options:
        reasons.append("exact_options_mismatch")

    if diagnostics is None:
        reasons.append("missing_diagnostics")
        diagnostics = {}
    if diagnostics.get("task_id") != task_id:
        reasons.append("diagnostics_task_id_mismatch")
    if diagnostics.get("reward") != trajectory.get("reward"):
        reasons.append("diagnostics_reward_mismatch")
    if (diagnostics.get("reward_detail") or {}) != reward_detail:
        reasons.append("diagnostics_reward_detail_mismatch")
    diagnostic_goal = diagnostics.get("goal_summary") or {}
    diagnostic_purchase = diagnostics.get("purchase_summary") or {}
    if not diagnostic_goal:
        reasons.append("missing_diagnostics_goal")
    elif not _summary_matches(
        goal,
        diagnostic_goal,
        ("asin", "name", "attributes", "goal_options", "price_upper"),
    ):
        reasons.append("diagnostics_goal_mismatch")
    if not diagnostic_purchase:
        reasons.append("missing_diagnostics_purchase")
    elif not _summary_matches(
        purchase,
        diagnostic_purchase,
        ("asin", "name", "attributes", "options", "price"),
    ):
        reasons.append("diagnostics_purchase_mismatch")
    if expected_model_name is not None and diagnostics.get("model_name") != expected_model_name:
        reasons.append("teacher_model_mismatch")
    if diagnostics.get("thinking") != "enabled":
        reasons.append("teacher_thinking_not_enabled")
    if expected_run_name is not None and diagnostics.get("run_name") != expected_run_name:
        reasons.append("diagnostics_rollout_mismatch")
    if diagnostics.get("termination_reason") != "purchase" or not diagnostics.get(
        "completed_purchase", False
    ):
        reasons.append("not_completed_by_purchase")
    if int(diagnostics.get("invalid_tool_response_count", 0) or 0) != 0:
        reasons.append("hidden_tool_retry")
    if diagnostics.get("tool_errors"):
        reasons.append("tool_error")

    messages = trajectory.get("conversation")
    actions: List[Dict[str, Any]] = []
    seen_state_actions = set()
    tool_ids = set()
    if not isinstance(messages, list) or not messages:
        reasons.append("missing_conversation")
        messages = []

    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    expected_roles = ["system", "user"]
    if len(messages) >= 2:
        expected_roles.extend(
            "assistant" if index % 2 == 0 else "tool"
            for index in range(2, len(messages))
        )
    if roles != expected_roles or not roles or roles[-1] != "tool":
        reasons.append("invalid_conversation_role_sequence")
    if roles.count("system") != 1 or roles.count("user") != 1:
        reasons.append("not_single_turn_conversation")

    if messages and isinstance(messages[0], dict):
        system_content = str(messages[0].get("content") or "")
        system_hash = hashlib.sha256(system_content.encode("utf-8")).hexdigest()
        if (
            expected_system_prompt_sha256 is not None
            and system_hash != expected_system_prompt_sha256
        ):
            reasons.append("system_prompt_mismatch")
    if len(messages) > 1 and isinstance(messages[1], dict):
        if (
            expected_instruction_sha256 is not None
            and _instruction_sha256(str(messages[1].get("content") or ""))
            != expected_instruction_sha256
        ):
            reasons.append("instruction_task_mismatch")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            reasons.append("non_object_message")
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) != 1:
            reasons.append("assistant_not_exactly_one_tool")
            continue
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            reasons.append("invalid_tool_call_object")
            continue
        call_id = tool_call.get("id")
        function = tool_call.get("function") or {}
        name = function.get("name")
        if not isinstance(call_id, str) or not call_id:
            reasons.append("missing_tool_call_id")
        elif call_id in tool_ids:
            reasons.append("duplicate_tool_call_id")
        else:
            tool_ids.add(call_id)
        if name not in {"search", "click"}:
            reasons.append("unsupported_tool")
            continue
        try:
            arguments = parse_arguments(tool_call)
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("invalid_tool_arguments")
            continue

        required_key = "keywords" if name == "search" else "value"
        argument = arguments.get(required_key)
        if not isinstance(argument, str) or not argument.strip():
            reasons.append("missing_required_tool_argument")
            continue

        if index == 0 or messages[index - 1].get("role") not in {"user", "tool"}:
            reasons.append("missing_preceding_state")
            state_content = ""
        else:
            state_content = str(messages[index - 1].get("content") or "")
        search_available, buttons = _state_controls(state_content)
        if search_available is None or buttons is None:
            reasons.append("unparseable_state_controls")
        elif name == "search" and not search_available:
            reasons.append("search_when_unavailable")
        elif name == "click":
            normalized_buttons = {normalize_click_value(button) for button in buttons}
            if normalize_click_value(argument) not in normalized_buttons:
                reasons.append("click_not_available")

        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            reasons.append("missing_tool_response")
        elif messages[index + 1].get("tool_call_id") != call_id:
            reasons.append("tool_call_id_mismatch")

        action = {"tool": name, "arguments": arguments}
        actions.append(action)
        state_hash = hashlib.sha256(state_content.encode("utf-8")).hexdigest()
        signature = json.dumps(action, ensure_ascii=False, sort_keys=True)
        state_action = (state_hash, signature)
        if state_action in seen_state_actions:
            reasons.append("repeated_state_action_loop")
        seen_state_actions.add(state_action)

    if not actions:
        reasons.append("no_actions")

    for message in messages:
        if message.get("role") not in {"system", "user"}:
            continue
        content = str(message.get("content") or "")
        if any(marker in content for marker in LEAK_MARKERS):
            reasons.append("privileged_label_leak")
            break

    for message in messages[:-1]:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if "Reward Details [SEP]" in content or "Target [SEP] asin" in content:
            reasons.append("privileged_observation_leak")
            break

    action_signatures = [
        json.dumps(action, ensure_ascii=False, sort_keys=True) for action in actions
    ]
    usage = diagnostics.get("model_usage") or {}
    metrics = {
        "task_id": task_id,
        "num_tool_calls": len(actions),
        "repeated_action_count": len(action_signatures) - len(set(action_signatures)),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "exact_asin_match": purchase.get("asin") == goal.get("asin"),
        "exact_options_match": expected_options == actual_options,
    }
    unique_reasons = list(dict.fromkeys(reasons))
    return VerificationResult(not unique_reasons, unique_reasons, metrics)


def load_and_verify(
    trajectory_path: Path,
    diagnostics_path: Path,
    train_start: int,
    train_end: int,
    **expected: Any,
) -> VerificationResult:
    with trajectory_path.open(encoding="utf-8") as handle:
        trajectory = json.load(handle)
    diagnostics = None
    if diagnostics_path.exists():
        with diagnostics_path.open(encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    return verify_trajectory(
        trajectory, diagnostics, train_start, train_end, **expected
    )
