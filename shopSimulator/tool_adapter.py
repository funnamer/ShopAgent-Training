"""Translate standard OpenAI tool calls to the unchanged ShopEnv API."""

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


CLICKABLES_RE = re.compile(r"可点击的按钮:\s*(\[[^\n]*\])")


SHOP_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the shopping catalog when search is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Precise product search keywords.",
                    }
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click one exact value from the current clickable-buttons list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Exact value from the current clickable-buttons list.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
]

ASK_SHOPPER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_shopper",
        "description": "Ask the shopper one question to clarify their requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A natural-language question for the shopper.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ToolExecution:
    """One executed tool call and its standard role=tool response."""

    name: str
    output: Dict[str, Any]
    message: Dict[str, Any]


def assistant_message_to_dict(message: Any) -> Dict[str, Any]:
    """Convert an OpenAI SDK message, or compatible dict, for message history."""
    if isinstance(message, dict):
        result = dict(message)
    elif hasattr(message, "model_dump"):
        result = message.model_dump(exclude_none=True)
        # OpenAI-compatible providers may return provider-specific fields in
        # model_extra. DeepSeek requires reasoning_content from every previous
        # assistant turn to be sent back when thinking mode uses tools.
        model_extra = getattr(message, "model_extra", None) or {}
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = model_extra.get("reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
    else:
        raise TypeError(f"Unsupported assistant message type: {type(message)!r}")
    result.setdefault("role", "assistant")
    return result


class ShopToolAdapter:
    """Execute model tool calls without changing the environment implementation."""

    def __init__(
        self,
        shop_env: Any,
        ask_shopper: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.shop_env = shop_env
        self.ask_shopper = ask_shopper
        self.current_observation: Optional[str] = None
        self.current_clickables: Optional[List[str]] = None
        self.last_executed_state_action: Optional[Tuple[str, str]] = None

    def update_observation(self, observation: str) -> None:
        """Record the latest environment state used to ground future clicks."""
        self.current_observation = observation
        self.current_clickables = self._parse_clickables(observation)

    @property
    def tools(self) -> List[Dict[str, Any]]:
        tools = list(SHOP_TOOLS)
        if self.ask_shopper is not None:
            tools.append(ASK_SHOPPER_TOOL)
        return tools

    def execute_assistant_message(self, message: Dict[str, Any]) -> ToolExecution:
        """Validate and execute exactly one standard Chat Completions tool call."""
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise ValueError(
                f"Agent must make exactly one tool call per turn; got {len(tool_calls)}"
            )

        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            if hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump(exclude_none=True)
            else:
                raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")

        function = tool_call.get("function") or {}
        name = function.get("name")
        call_id = tool_call.get("id")
        if not name or not call_id:
            raise ValueError("Tool call is missing function.name or id")

        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                raw_arguments
                if isinstance(raw_arguments, dict)
                else json.loads(raw_arguments)
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON arguments for tool {name}") from exc

        output = self._execute(name, arguments)
        return ToolExecution(
            name=name,
            output=output,
            message={
                "role": "tool",
                "tool_call_id": call_id,
                # Only feed the model the observation it needs for its next
                # decision.  Structured execution metadata remains available
                # through ToolExecution.output for control flow/diagnostics.
                "content": self._message_content(output),
            },
        )

    @staticmethod
    def _message_content(output: Dict[str, Any]) -> str:
        if output.get("source") == "environment":
            if output.get("ok", False):
                return str((output.get("result") or {}).get("instruction", ""))
            return str(output.get("error", "Environment error"))
        if output.get("source") == "shopper":
            return str(output.get("response", ""))
        if output.get("source") == "validation":
            return str(output.get("error", "Invalid tool action"))
        return ""

    def _execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "search":
            keywords = self._required_string(arguments, "keywords", name)
            return self._execute_once_per_state(f"search[{keywords}]")
        if name == "click":
            value = self._required_string(arguments, "value", name)
            canonical_value = self._canonical_clickable(value)
            if self.current_clickables is not None and canonical_value is None:
                return self._validation_error(
                    f"无效点击：{value} 不在当前可点击按钮中。"
                    f"当前只能选择 {json.dumps(self.current_clickables, ensure_ascii=False)}。"
                    "请重新调用 click，并且 value 必须与其中一个按钮完全对应。",
                    requested_action=f"click[{value}]",
                )

            resolved_value = canonical_value if canonical_value is not None else value
            action = f"click[{resolved_value}]"
            return self._execute_once_per_state(action)
        if name == "ask_shopper" and self.ask_shopper is not None:
            question = self._required_string(arguments, "question", name)
            return {
                "ok": True,
                "source": "shopper",
                "response": self.ask_shopper(question),
            }
        raise ValueError(f"Unsupported tool: {name}")

    def _canonical_clickable(self, value: str) -> Optional[str]:
        if self.current_clickables is None:
            return None
        normalized_value = self._normalize_click_value(value)
        for clickable in self.current_clickables:
            if self._normalize_click_value(clickable) == normalized_value:
                return clickable
        return None

    @staticmethod
    def _normalize_click_value(value: str) -> str:
        """Match ShopEnv's case-insensitive clickable lookup semantics."""
        return unicodedata.normalize("NFKC", value).strip().casefold()

    @staticmethod
    def _parse_clickables(observation: str) -> Optional[List[str]]:
        matches = CLICKABLES_RE.findall(observation)
        if not matches:
            return None
        try:
            parsed = json.loads(matches[-1])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            return None
        return parsed

    def _validation_error(
        self, error: str, requested_action: str
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "recoverable": True,
            "source": "validation",
            "action": requested_action,
            "available_clickables": self.current_clickables,
            "error": error,
        }

    def _execute_once_per_state(self, action: str) -> Dict[str, Any]:
        state_action = (self.current_observation or "", action)
        if self.last_executed_state_action == state_action:
            return self._validation_error(
                f"重复动作：{action} 已在完全相同的页面状态执行过，页面没有变化。"
                "请重新检查当前可用操作，并选择不同的有效操作。",
                requested_action=action,
            )
        self.last_executed_state_action = state_action
        return self._execute_env(action)

    def _execute_env(self, action: str) -> Dict[str, Any]:
        result = self.shop_env.interact(action)
        if "error" in result:
            return {
                "ok": False,
                "source": "environment",
                "action": action,
                "error": result["error"],
            }
        return {
            "ok": True,
            "source": "environment",
            "action": action,
            "result": result,
        }

    @staticmethod
    def _required_string(arguments: Dict[str, Any], key: str, tool_name: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Tool {tool_name} requires a non-empty string '{key}'")
        return value.strip()
