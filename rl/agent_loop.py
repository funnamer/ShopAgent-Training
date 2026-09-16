"""Minimal verl ToolAgentLoop extension for ShopSimulator lifecycle handling."""

from __future__ import annotations

from uuid import uuid4

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from rl.session import CURRENT_ROLLOUT, REGISTRY


class ShopAgentLoop(ToolAgentLoop):
    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        if state is not AgentState.TERMINATED:
            return state

        key = CURRENT_ROLLOUT.get()
        session = REGISTRY.get(key) if key else None
        if session is None or session.trace.get("termination"):
            return state

        if len(agent_data.response_mask) >= self.response_length:
            session.trace["termination"] = "context_limit"
        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            session.trace["termination"] = "turn_limit"
        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            session.trace["termination"] = "turn_limit"
        else:
            session.trace["termination"] = "invalid_model"
        return state

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        key = CURRENT_ROLLOUT.get()
        session = REGISTRY.get(key) if key else None
        if session is not None and session.trace.get("termination"):
            return AgentState.TERMINATED
        return state

    async def run(self, sampling_params: dict, **kwargs):
        key = uuid4().hex
        token = CURRENT_ROLLOUT.set(key)
        output = None
        try:
            output = await super().run(sampling_params, **kwargs)
            return output
        finally:
            trace = await REGISTRY.finalize(key, default_reason="invalid_model")
            if output is not None:
                if trace is not None:
                    # verl's async reward loop forwards every AgentLoop
                    # extra field through ``tool_extra_fields`` and merges it
                    # into ``extra_info`` before calling compute_score.
                    # Do not emit an ``extra_info`` field here: the original
                    # dataset batch already owns that key, and DataProto.union
                    # requires duplicate non-tensor fields to be identical.
                    output.extra_fields["shop_trace"] = trace
            CURRENT_ROLLOUT.reset(token)
