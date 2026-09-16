from __future__ import annotations

import unittest

from rl.session import MAX_VALIDATION_RETRIES, ShopSession, SessionRegistry
from tool_adapter import ShopToolAdapter


class _FakeEnv:
    env_idx = 1

    def __init__(self, result=None):
        self.result = result or {"instruction": "next", "done": False, "over": False}
        self.calls = []

    def interact(self, action):
        self.calls.append(action)
        return dict(self.result)


def _session(env, observation='可点击的按钮: ["ok"]'):
    adapter = ShopToolAdapter(env)
    adapter.update_observation(observation)
    return ShopSession(
        key="test",
        task_id=1,
        env=env,
        adapter=adapter,
        trace={
            "task_id": 1,
            "termination": None,
            "seen_asins": [],
            "actions": [],
            "action_count": 0,
            "repeat_count": 0,
            "invalid_count": 0,
            "validation_retry_count": 0,
            "back_count": 0,
        },
    )


class SessionRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_clicks_get_same_three_retries_as_eval_agent(self):
        session = _session(_FakeEnv())
        registry = SessionRegistry()

        for _ in range(MAX_VALIDATION_RETRIES):
            _, metrics = await registry.execute(session, "click", {"value": "bad"})
            self.assertTrue(metrics["invalid"])
            self.assertNotIn("terminal", metrics)

        _, metrics = await registry.execute(session, "click", {"value": "bad"})
        self.assertTrue(metrics["terminal"])
        self.assertEqual(session.trace["termination"], "invalid_model")

    async def test_valid_action_resets_consecutive_validation_retries(self):
        env = _FakeEnv()
        session = _session(env)
        registry = SessionRegistry()

        await registry.execute(session, "click", {"value": "bad"})
        await registry.execute(session, "click", {"value": "ok"})

        self.assertEqual(session.trace["validation_retry_count"], 0)
        self.assertEqual(env.calls, ["click[ok]"])

    async def test_nonrecoverable_adapter_error_terminates(self):
        env = _FakeEnv({"error": "environment rejected action"})
        session = _session(env, observation="no clickable metadata")
        registry = SessionRegistry()

        _, metrics = await registry.execute(session, "search", {"keywords": "x"})

        self.assertTrue(metrics["terminal"])
        self.assertEqual(session.trace["termination"], "invalid_model")


if __name__ == "__main__":
    unittest.main()
