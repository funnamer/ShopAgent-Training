"""verl BaseTool bridge that reuses ShopEnv and ShopToolAdapter."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from rl.session import CURRENT_ROLLOUT, REGISTRY


class ShopSimulatorTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.base_url = str(config.get("base_url", "http://127.0.0.1:5000")).rstrip("/")

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> tuple[str, ToolResponse]:
        # verl creates/releases a BaseTool lifecycle for each call. A shopping
        # session spans several calls, so its state is owned by ShopAgentLoop's
        # rollout-scoped registry instead of this per-call instance id.
        return instance_id or uuid4().hex, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise RuntimeError("ShopSimulatorTool requires verl agent_data")
        rollout_key = CURRENT_ROLLOUT.get()
        if not rollout_key:
            raise RuntimeError("ShopSimulatorTool called outside ShopAgentLoop")

        tool_kwargs = agent_data.tools_kwargs.get(self.name, {})
        create_kwargs = tool_kwargs.get("create_kwargs", {})
        task_id = int(create_kwargs["task_id"])
        session = await REGISTRY.ensure(rollout_key, task_id, self.base_url)
        if session.trace.get("termination") == "infrastructure_error":
            return ToolResponse(text=session.trace.get("error", "environment error")), 0.0, {
                "infrastructure_error": True
            }

        text, metrics = await REGISTRY.execute(session, self.name, parameters)
        return ToolResponse(text=text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        # Intentional no-op; ShopAgentLoop owns rollout-scoped cleanup.
        del instance_id, kwargs
