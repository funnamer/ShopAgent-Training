"""Agent module: implements main logic for shopping agent."""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_adapter import ShopToolAdapter, assistant_message_to_dict
from task_selection import select_task_ids, task_selection_metadata

DEFAULT_MAX_TOKENS = 512
DEFAULT_THINKING_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 200
DEFAULT_MAX_VALIDATION_RETRIES = 3
IDEALAB_DEFAULT_KEY = "{your_api_key}"  # Should be set via config file or environment variable
IDEALAB_DEFAULT_BASE_URL = "{your_base_url}"  # Should be set via config file or environment variable
FAILED_CALL_MESSAGE = "failed call"


def load_env_file() -> None:
    """Load the repository .env without overriding exported variables."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


class Agent:
    """Shopping agent class responsible for interacting with shopping environment."""

    def __init__(self, task_id: int, config: Dict[str, Any]) -> None:
        """
        Initialize Agent.

        Args:
            task_id: Task ID
            config: Dictionary containing model config, API config, etc.
        """
        self.task_id = task_id
        self.config = config
        self.model_name = self.config["model_name"]
        self.run_name = self.config.get("run_name", self.model_name)
        self.source = config["source"]
        self.thinking = self.config.get(
            "thinking", "enabled" if self.source == "deepseek" else None
        )
        if self.source == "deepseek" and self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DeepSeek thinking must be 'enabled' or 'disabled'")
        self.tool_choice = self.config.get(
            "tool_choice", "auto" if self.thinking == "enabled" else "required"
        )
        if self.source == "deepseek" and self.thinking == "enabled":
            if self.tool_choice != "auto":
                raise ValueError(
                    "DeepSeek thinking mode requires tool_choice='auto'; "
                    "'required' and named tool choices return HTTP 400"
                )

        if self.source == "idealab":
            default_key = IDEALAB_DEFAULT_KEY
            default_base_url = IDEALAB_DEFAULT_BASE_URL
        else:
            default_key = None
            default_base_url = None

        key_env = self.config.get("model_key_env")
        base_url_env = self.config.get("base_url_env")
        self.model_key = (
            os.getenv(key_env) if key_env else self.config.get("model_key", default_key)
        )
        self.base_url = (
            os.getenv(base_url_env)
            if base_url_env
            else self.config.get("base_url", default_base_url)
        )

        compatible_api_config = {
            source_name: {
                "api_key": self.model_key,
                "model_name": self.model_name,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
            }
            for source_name in ("idealab", "openai", "deepseek")
        }
        if self.source not in compatible_api_config:
            raise ValueError(f"Unsupported data source: {self.source}")

        self.api_config = compatible_api_config[self.source]
        self.system_prompt = config["system_prompt"]

        self.shop_env: Optional[ShopEnv] = None
        self.task_complete = False
        self.conversation_log: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        self.user_persona: Optional[Dict[str, Any]] = None
        self.env_idx: Optional[int] = None
        self.tool_adapter: Optional[ShopToolAdapter] = None
        self.initial_instruction = ""
        self.last_valid_observation = ""
        self.last_env_response: Dict[str, Any] = {}
        self.tool_outputs: List[Dict[str, Any]] = []
        self.usage_by_call: List[Dict[str, Any]] = []
        self.invalid_tool_response_count = 0
        self.validation_retry_count = 0
        self.diagnostic_saved = False

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        """
        Set ShopEnv instance.

        Args:
            shop_env: ShopEnv instance
        """
        self.shop_env = shop_env
        self.tool_adapter = ShopToolAdapter(shop_env)

    def reset(self) -> str:
        """
        Reset state and start new task.

        Returns:
            str: Initial instruction

        Raises:
            AttributeError: When shop_env is not initialized
        """
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx
        instruction = env_result["instruction"] + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"

        self.task_complete = False
        self.conversation_log = []
        self.initial_instruction = instruction
        self.last_valid_observation = instruction
        self.last_env_response = {}
        self.tool_outputs = []
        self.usage_by_call = []
        self.invalid_tool_response_count = 0
        self.validation_retry_count = 0
        self.diagnostic_saved = False

        if self.tool_adapter is not None:
            self.tool_adapter.update_observation(instruction)

        if self.shop_env.if_persona and "user_persona" in env_result:
            self.user_persona = env_result["user_persona"]
            persona_str = json.dumps(self.user_persona, ensure_ascii=False)
            persona_content = f"\n用户的个人文档是：{persona_str}"
            self.messages = [
                {"role": "system", "content": self.system_prompt + persona_content}
            ]
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        return instruction

    def act(self, instruction: str) -> Tuple[bool, str]:
        """
        Execute one interaction step.

        Args:
            instruction: Instruction content

        Returns:
            Tuple[bool, str]: (task_complete, observation)
                - task_complete: Whether task is complete
                - observation: Observation result

        Raises:
            AttributeError: When shop_env is not initialized
            ValueError: When environment error occurs
            RuntimeError: When LLM call fails
        """
        if self.shop_env is None or self.tool_adapter is None:
            raise AttributeError(
                "Agent not correctly connected to ShopEnv, please check the configuration."
            )

        # The initial observation is a user message. Subsequent observations are
        # already in the history as standard role=tool results.
        if not self.messages or self.messages[-1].get("role") != "tool":
            self.messages.append({"role": "user", "content": instruction})

        assistant_message = self.get_response_idealab(self.messages)

        if assistant_message == FAILED_CALL_MESSAGE:
            raise RuntimeError("LLM call failed")

        self.messages.append(assistant_message)
        execution = self.tool_adapter.execute_assistant_message(assistant_message)
        self.messages.append(execution.message)
        self.tool_outputs.append(execution.output)
        if not execution.output.get("ok", False):
            if execution.output.get("recoverable", False):
                self.validation_retry_count += 1
                max_validation_retries = self.config.get(
                    "max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES
                )
                if self.validation_retry_count <= max_validation_retries:
                    return False, execution.message["content"]
                raise ValueError(
                    "工具动作连续校验失败，已超过最大纠错次数 "
                    f"({max_validation_retries}): "
                    f"{execution.output.get('error', 'unknown validation error')}"
                )
            raise ValueError(f"环境报错: {execution.output.get('error', 'unknown error')}")
        self.validation_retry_count = 0
        env_response = execution.output["result"]

        observation = env_response.get("instruction", "")
        self.last_env_response = env_response
        if observation:
            self.last_valid_observation = observation
            self.tool_adapter.update_observation(observation)

        if env_response.get("done", False) or env_response.get("over", False):
            reward = env_response.get("reward", 0)
            reward_detail = env_response.get("reward_detail", {})
            goal = env_response.get("goal", {})
            purchase = env_response.get("purchase", {})
            termination_reason = (
                "purchase" if env_response.get("done", False) else "history_limit"
            )

            self.save_to_json(reward, reward_detail, goal, purchase)
            self.save_diagnostics(
                termination_reason=termination_reason,
                reward=reward,
                reward_detail=reward_detail,
                goal=goal,
                purchase=purchase,
            )
            self.task_complete = True

        return self.task_complete, observation

    def get_response_idealab(
        self, messages: List[Dict[str, Any]], max_try: int = DEFAULT_MAX_RETRY
    ) -> Any:
        """
        Call idealab LLM API.

        Args:
            messages: Message list
            max_try: Maximum retry attempts

        Returns:
            str: LLM response content, returns "failed call" on failure
        """
        if self.model_key is None:
            raise ValueError("model_key not set, cannot call API")
        if self.base_url is None:
            raise ValueError("base_url not set, cannot call API")

        client = OpenAI(
            api_key=self.model_key,
            base_url=self.base_url,
        )

        request_messages = messages
        tool_call_retry_used = False
        for attempt in range(max_try):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": request_messages,
                    "tools": self.tool_adapter.tools if self.tool_adapter else [],
                    "tool_choice": self.tool_choice,
                    "max_tokens": self.config.get(
                        "max_tokens",
                        DEFAULT_THINKING_MAX_TOKENS
                        if self.thinking == "enabled"
                        else DEFAULT_MAX_TOKENS,
                    ),
                    "stream": False,
                }
                thinking = self.thinking
                if thinking:
                    request_kwargs["extra_body"] = {
                        "thinking": {"type": thinking}
                    }
                if thinking == "enabled":
                    request_kwargs["reasoning_effort"] = self.config.get(
                        "reasoning_effort", "high"
                    )
                else:
                    request_kwargs["temperature"] = self.config.get(
                        "temperature", DEFAULT_TEMPERATURE
                    )
                completion = client.chat.completions.create(
                    **request_kwargs,
                )
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    if hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump(exclude_none=True)
                    elif isinstance(usage, dict):
                        usage_dict = dict(usage)
                    else:
                        usage_dict = {}
                    if usage_dict:
                        self.usage_by_call.append(usage_dict)
                assistant_message = assistant_message_to_dict(
                    completion.choices[0].message
                )
                tool_calls = assistant_message.get("tool_calls") or []
                if self.thinking == "enabled" and len(tool_calls) != 1:
                    self.invalid_tool_response_count += 1
                    if not tool_call_retry_used:
                        tool_call_retry_used = True
                        # Thinking mode only permits tool_choice=auto. If the
                        # model answers with text or emits multiple tools, keep
                        # its reasoning_content in the corrective request as
                        # required by DeepSeek's tool-call protocol.
                        request_messages = [
                            *messages,
                            assistant_message,
                            {
                                "role": "system",
                                "content": (
                                    "当前购物任务尚未完成。请不要输出普通文本；"
                                    "必须从当前可用操作中调用且只调用一个工具。"
                                ),
                            },
                        ]
                        continue
                return assistant_message
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_try}): {e}")
                status_code = getattr(e, "status_code", None)
                non_retryable = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code != 429
                )
                if non_retryable or attempt == max_try - 1:
                    traceback.print_exc()
                if non_retryable:
                    return FAILED_CALL_MESSAGE
                continue

        return FAILED_CALL_MESSAGE

    def save_to_json(
        self,
        reward: float,
        reward_detail: Dict[str, Any],
        goal: Dict[str, Any],
        purchase: Dict[str, Any],
    ) -> None:
        """
        Save conversation and scores to JSON file.

        Args:
            reward: Reward score
            reward_detail: Reward details
            goal: Goal information
            purchase: Purchase information
        """
        log_data = {
            "task_id": self.task_id,
            "reward": reward,
            "reward_detail": reward_detail,
            "goal": goal,
            "purchase": purchase,
            "conversation": self.messages,
        }

        file_path = os.path.join(self.config["output_path"], self.run_name)
        os.makedirs(file_path, exist_ok=True)

        filename = os.path.join(file_path, f"{self.task_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        print(f"[LOG] Saved to file: {filename}")

    def save_diagnostics(
        self,
        termination_reason: str,
        reward: Optional[float] = None,
        reward_detail: Optional[Dict[str, Any]] = None,
        goal: Optional[Dict[str, Any]] = None,
        purchase: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Save a compact, separate record for trajectory diagnosis."""
        if self.diagnostic_saved:
            return

        actions = []
        tool_errors = [
            output for output in self.tool_outputs if output.get("ok") is False
        ]
        for message in self.messages:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    raw_arguments = function.get("arguments") or "{}"
                    try:
                        arguments = (
                            raw_arguments
                            if isinstance(raw_arguments, dict)
                            else json.loads(raw_arguments)
                        )
                    except (TypeError, json.JSONDecodeError):
                        arguments = {"_raw": raw_arguments}
                    actions.append(
                        {
                            "step": len(actions) + 1,
                            "tool": function.get("name"),
                            "arguments": arguments,
                        }
                    )
        search_actions = [action for action in actions if action["tool"] == "search"]
        click_actions = [action for action in actions if action["tool"] == "click"]
        click_values = [
            action["arguments"].get("value")
            for action in click_actions
            if isinstance(action["arguments"], dict)
        ]
        visited_asins = list(
            dict.fromkeys(
                value
                for value in click_values
                if isinstance(value, str) and value.isdigit()
            )
        )
        action_signatures = {
            json.dumps(
                {"tool": action["tool"], "arguments": action["arguments"]},
                ensure_ascii=False,
                sort_keys=True,
            )
            for action in actions
        }

        token_usage = {
            "api_calls_with_usage": len(self.usage_by_call),
            "prompt_tokens": sum(
                usage.get("prompt_tokens", 0) for usage in self.usage_by_call
            ),
            "completion_tokens": sum(
                usage.get("completion_tokens", 0) for usage in self.usage_by_call
            ),
            "total_tokens": sum(
                usage.get("total_tokens", 0) for usage in self.usage_by_call
            ),
            "by_call": self.usage_by_call,
        }

        purchase = purchase or {}
        goal = goal or {}
        diagnostic = {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "run_name": self.run_name,
            "thinking": self.thinking or "disabled",
            "reasoning_effort": self.config.get("reasoning_effort"),
            "env_idx": self.env_idx,
            "termination_reason": termination_reason,
            "completed_purchase": termination_reason == "purchase",
            "error": error,
            "reward": 0 if reward is None else reward,
            "reward_detail": reward_detail or {},
            "instruction": self.initial_instruction,
            "num_tool_calls": len(actions),
            "invalid_tool_response_count": self.invalid_tool_response_count,
            "num_searches": len(search_actions),
            "num_clicks": len(click_actions),
            "num_back_to_search": click_values.count("back to search"),
            "num_detail_views": sum(
                value in {"description", "features", "reviews"}
                for value in click_values
            ),
            "repeated_action_count": len(actions) - len(action_signatures),
            "visited_asins": visited_asins,
            "selected_options": purchase.get("options", {}),
            "last_action": actions[-1] if actions else None,
            "actions": actions,
            "tool_errors": tool_errors,
            "model_usage": token_usage,
            "goal_summary": {
                key: goal.get(key)
                for key in (
                    "asin",
                    "name",
                    "attributes",
                    "goal_options",
                    "price_upper",
                )
                if key in goal
            },
            "purchase_summary": {
                key: purchase.get(key)
                for key in ("asin", "name", "attributes", "options", "price")
                if key in purchase
            },
            "last_valid_observation": self.last_valid_observation,
        }

        output_root = Path(self.config["output_path"]) / self.run_name
        diagnostics_dir = output_root / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_file = diagnostics_dir / f"{self.task_id}.json"
        with diagnostics_file.open("w", encoding="utf-8") as handle:
            json.dump(diagnostic, handle, ensure_ascii=False, indent=4)

        self.diagnostic_saved = True
        print(f"[DIAGNOSTIC] Saved to file: {diagnostics_file}")


def get_finished_task(out_path: str) -> List[int]:
    """
    Get list of completed tasks.

    Args:
        out_path: Output directory path

    Returns:
        List[int]: List of completed task IDs
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    json_files = []
    for filename in os.listdir(out_path):
        if filename.endswith(".json"):
            task_id_str = filename.rsplit(".", 1)[0]
            try:
                json_files.append(int(task_id_str))
            except ValueError:
                print(
                    f"Warning: Unable to parse task ID '{task_id_str}', skipping file"
                )

    return sorted(json_files)


def run_task(task_id: int, config: Dict[str, Any]) -> None:
    """
    Run single task.

    Args:
        task_id: Task ID
        config: Configuration dictionary
    """
    agent: Optional[Agent] = None
    shop_env: Optional[ShopEnv] = None

    try:
        shop_env = ShopEnv(config["env_config"])
        agent = Agent(task_id=task_id, config=config["agent_config"])
        agent.set_shop_env(shop_env)
        observation = agent.reset()

        while True:
            done, observation = agent.act(observation)
            if done:
                break
    except Exception as e:
        print(f"Error (task {task_id}): {e}")
        traceback.print_exc()
        if agent is not None:
            try:
                agent.save_diagnostics(
                    termination_reason="error",
                    error=f"{type(e).__name__}: {e}",
                )
            except Exception:
                traceback.print_exc()
    finally:
        if agent is not None and agent.shop_env is not None:
            agent.shop_env.release()


def run_tasks_multithreaded(
    todo_tasks: List[int], config: Dict[str, Any], max_workers: int = 4
) -> None:
    """
    Execute tasks using multithreading.

    Args:
        todo_tasks: List of task IDs to process
        config: Configuration dictionary
        max_workers: Maximum number of worker threads
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_task, task_id, config): task_id
            for task_id in todo_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(todo_tasks),
            desc="执行任务",
        ):
            task_id = future_to_task[future]
            try:
                future.result()
                print(f"**Task {task_id} completed**")
            except Exception as e:
                print(f"**Task {task_id} failed**: {e}")


def run_tasks_singlethreaded(
    todo_tasks: List[int], config: Dict[str, Any]
) -> None:
    """
    Execute tasks in single-threaded mode.

    Args:
        todo_tasks: List of task IDs to process
        config: Configuration dictionary
    """
    for task_id in tqdm(todo_tasks, desc="执行任务"):
        print(f"**Begin task {task_id}**")
        run_task(task_id, config)


def main() -> None:
    """Main function: parse arguments and execute tasks."""
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Batch process tasks from YAML config files."
    )
    parser.add_argument(
        "--yaml_name", required=True, help="YAML config file name"
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Whether to use multithreading to execute",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads in multithreaded mode",
    )
    args = parser.parse_args()

    config_file = args.yaml_name
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found '{config_file}'")
        return
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing failed: {e}")
        return

    agent_config = config.get("agent_config")
    if agent_config is None:
        print("Error: 'agent_config' missing in config file")
        return

    output_path = os.path.join(
        agent_config["output_path"],
        agent_config.get("run_name", agent_config["model_name"]),
    )
    try:
        selected_tasks = select_task_ids(agent_config)
    except ValueError as e:
        print(f"Error: Invalid task selection config: {e}")
        return

    finished_tasks = get_finished_task(output_path)
    finished_task_set = set(finished_tasks)
    selected_finished = [i for i in selected_tasks if i in finished_task_set]
    todo_tasks = [i for i in selected_tasks if i not in finished_task_set]

    metadata_dir = os.path.join(output_path, "run_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(
        os.path.join(metadata_dir, "task_selection.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            task_selection_metadata(agent_config, selected_tasks),
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"Task pool: {agent_config.get('task_nums', 0)}, "
        f"Selected: {len(selected_tasks)}, "
        f"Completed in selection: {len(selected_finished)}, "
        f"Pending: {len(todo_tasks)}"
    )
    print(f"Selected task IDs: {selected_tasks}")

    if args.multithread:
        print(f"Using multithreaded mode, max workers: {args.max_workers}")
        run_tasks_multithreaded(todo_tasks, config, args.max_workers)
    else:
        print("Using single-threaded mode")
        run_tasks_singlethreaded(todo_tasks, config)


if __name__ == "__main__":
    main()
