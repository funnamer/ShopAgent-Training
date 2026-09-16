"""Rollout-scoped ShopSimulator sessions shared by the verl tools and loop."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from single_eval.env import ShopEnv
from tool_adapter import ShopToolAdapter


CURRENT_ROLLOUT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "shopsim_grpo_rollout", default=None
)
ASIN_RE = re.compile(r"(?<!\d)\d{10,20}(?!\d)")
MAX_VALIDATION_RETRIES = 3


@dataclass
class ShopSession:
    key: str
    task_id: int
    env: ShopEnv
    adapter: ShopToolAdapter
    trace: dict[str, Any] = field(default_factory=dict)
    released: bool = False


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, ShopSession] = {}
        self._lock = asyncio.Lock()

    async def ensure(self, key: str, task_id: int, base_url: str) -> ShopSession:
        async with self._lock:
            current = self._sessions.get(key)
            if current is not None:
                if current.task_id != task_id:
                    raise RuntimeError(f"rollout {key} changed task_id from {current.task_id} to {task_id}")
                return current

        # Each rollout key is unique, so the expensive HTTP reset can happen
        # outside the global lock and all 16 sessions initialize concurrently.
        env = ShopEnv({"base_url": base_url, "if_persona": False})
        try:
            reset_result = await asyncio.to_thread(env.reset, task_id)
            observation = (
                reset_result["instruction"]
                + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
            )
            adapter = ShopToolAdapter(env)
            adapter.update_observation(observation)
            trace = {
                "task_id": task_id,
                "termination": None,
                "seen_asins": [],
                "actions": [],
                "action_count": 0,
                "repeat_count": 0,
                "invalid_count": 0,
                "validation_retry_count": 0,
                "back_count": 0,
            }
        except Exception as exc:
            trace = {
                "task_id": task_id,
                "termination": "infrastructure_error",
                "error": repr(exc),
                "seen_asins": [],
                "actions": [],
                "action_count": 0,
                "repeat_count": 0,
                "invalid_count": 0,
                "validation_retry_count": 0,
                "back_count": 0,
            }
            adapter = ShopToolAdapter(env)
        session = ShopSession(key, task_id, env, adapter, trace)
        async with self._lock:
            self._sessions[key] = session
        return session

    def get(self, key: str) -> ShopSession | None:
        return self._sessions.get(key)

    async def execute(self, session: ShopSession, name: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if session.trace.get("termination"):
            return "任务已经结束，请勿继续操作。", {"terminal": True}

        call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": uuid4().hex,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(parameters, ensure_ascii=False)},
                }
            ],
        }
        try:
            execution = await asyncio.to_thread(session.adapter.execute_assistant_message, call)
        except (TypeError, ValueError) as exc:
            session.trace["invalid_count"] += 1
            session.trace["action_count"] += 1
            session.trace["actions"].append({"tool": name, "arguments": parameters, "valid": False})
            session.trace["termination"] = "invalid_model"
            return f"无效工具调用：{exc}。", {"invalid": True, "terminal": True}
        except Exception as exc:
            session.trace["termination"] = "infrastructure_error"
            session.trace["error"] = repr(exc)
            return f"ShopSimulator infrastructure error: {exc}", {"infrastructure_error": True}

        output = execution.output
        action_text = str(output.get("action") or f"{name}[{parameters}]")
        session.trace["action_count"] += 1
        session.trace["actions"].append(
            {"tool": name, "arguments": parameters, "action": action_text, "valid": bool(output.get("ok"))}
        )
        if not output.get("ok", False):
            session.trace["invalid_count"] += 1
            error = str(output.get("error") or "工具动作无效，请重新选择。")
            if "重复动作" in error:
                session.trace["repeat_count"] += 1
            if output.get("recoverable", False):
                session.trace["validation_retry_count"] += 1
                if session.trace["validation_retry_count"] <= MAX_VALIDATION_RETRIES:
                    return error, {"invalid": True}
                session.trace["termination"] = "invalid_model"
                return (
                    f"{error}\n连续校验失败已超过 {MAX_VALIDATION_RETRIES} 次，任务终止。",
                    {"invalid": True, "terminal": True},
                )
            session.trace["termination"] = "invalid_model"
            return error, {"invalid": True, "terminal": True}

        result = dict(output.get("result") or {})
        session.trace["validation_retry_count"] = 0
        observation = str(result.get("instruction") or "")
        if observation:
            session.adapter.update_observation(observation)
            for asin in ASIN_RE.findall(observation):
                if asin not in session.trace["seen_asins"]:
                    session.trace["seen_asins"].append(asin)

        normalized_action = action_text.casefold()
        if "back to search" in normalized_action or "< prev" in normalized_action:
            session.trace["back_count"] += 1

        if result.get("done"):
            session.trace["termination"] = "purchase"
            session.trace["final_result"] = result
        elif result.get("over"):
            session.trace["termination"] = "history_limit"
            session.trace["final_result"] = result

        return observation, {
            "terminal": bool(session.trace.get("termination")),
            "env_idx": session.env.env_idx,
        }

    async def finalize(self, key: str, default_reason: str = "timeout") -> dict[str, Any] | None:
        async with self._lock:
            session = self._sessions.pop(key, None)
        if session is None:
            return None
        if not session.trace.get("termination"):
            session.trace["termination"] = default_reason
        if not session.released and session.env.env_idx is not None:
            try:
                await asyncio.to_thread(session.env.release)
            except Exception as exc:
                if session.trace.get("termination") not in {"purchase", "history_limit"}:
                    session.trace["termination"] = "infrastructure_error"
                    session.trace["error"] = repr(exc)
            session.released = True
        return copy.deepcopy(session.trace)


REGISTRY = SessionRegistry()
