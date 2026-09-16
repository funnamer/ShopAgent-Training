"""Agent module: implements main logic for shopping agent."""

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv
from shopper import ShopperSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_adapter import ShopToolAdapter, assistant_message_to_dict
from task_selection import select_task_ids, task_selection_metadata

import pdb
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 50
IDEALAB_DEFAULT_KEY = "{your_api_key}"  # Should be set via config file or environment variable
IDEALAB_DEFAULT_BASE_URL = "{your_base_url}"  # Should be set via config file or environment variable

SEP_MARKER = "[SEP]"
ROLE_ENV = "Env"
ROLE_SHOPPER = "Shopper"
FAILED_CALL_MESSAGE = "failed call"

class Agent:
    """Shopping agent class responsible for interacting with shopping environment and shoppers."""

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
        self.source = config["source"]

        if self.source in ("idealab", "openai", "deepseek"):
            default_key = IDEALAB_DEFAULT_KEY
            default_base_url = IDEALAB_DEFAULT_BASE_URL
        else:
            default_key = None
            default_base_url = None
        
        key_env = self.config.get("model_key_env")
        base_url_env = self.config.get("base_url_env")
        self.model_key = os.getenv(key_env) if key_env else self.config.get("model_key", default_key)
        self.base_url = os.getenv(base_url_env) if base_url_env else self.config.get("base_url", default_base_url)

        api_config = {
            source_name: {
                "api_key": self.model_key,
                "model_name": self.model_name,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
            }
            for source_name in ("idealab", "openai", "deepseek")
        }
        if self.source not in api_config:
            raise ValueError(f"Unsupported data source: {self.source}")

        self.api_config = api_config[self.source]
        self.system_prompt = config["system_prompt"]
        self.agent_format_template = config.get("agent_format_template", "")

        self.shopper_simulator: Optional[ShopperSimulator] = None
        self.shop_env: Optional[ShopEnv] = None
        self.task_complete = False
        self.shopping_history: List[Dict[str, Any]] = []
        self.conversation_log: List[Dict[str, Any]] = []
        self.action_list: List[str] = []
        self.obs_list: List[str] = []
        self.messages: List[Dict[str, Any]] = []
        self.turn: int = 0
        self.max_turns: int = 0
        self.search_available: bool = True
        self.available_buttons: List[str] = []
        self.user_persona: Optional[Dict[str, Any]] = None
        self.env_idx: Optional[int] = None
        self.tool_adapter: Optional[ShopToolAdapter] = None

    def set_shopper(self, shopper_simulator: ShopperSimulator) -> None:
        """
        Set Shopper instance.

        Args:
            shopper_simulator: ShopperSimulator instance
        """
        self.shopper_simulator = shopper_simulator

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        """
        Set ShopEnv instance.

        Args:
            shop_env: ShopEnv instance
        """
        self.shop_env = shop_env
        self.tool_adapter = ShopToolAdapter(shop_env, ask_shopper=self._ask_shopper)

    def reset(self) -> Tuple[str, str]:
        """
        Reset state and start new task.

        Returns:
            Tuple[str, str]: (instruction, shopper_input) instruction and shopper input

        Raises:
            AttributeError: When shop_env or shopper_simulator is not initialized
        """
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx
        instruction = env_result["instruction"] + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        goal_options = env_result["goal_options"]

        if self.shopper_simulator is None:
            raise AttributeError("self.shopper_simulator not initialized")

        self.shopper_simulator.reset(instruction, goal_options)
        self.task_complete = False
        self.shopping_history = []
        self.conversation_log = []

        if self.shop_env.if_persona and "user_persona" in env_result:
            self.user_persona = env_result["user_persona"]
            persona_str = json.dumps(self.user_persona, ensure_ascii=False)
            persona_content = f"\n用户的个人文档是：{persona_str}"
            self.messages = [
                {"role": "system", "content": self.system_prompt + persona_content}
            ]
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        shopper_input = self._get_shopper_input()
        if not shopper_input:
            raise AttributeError("shopper response error")

        return instruction, shopper_input

    def process_observation(self, observation: str) -> str:
        """
        Process observation, extract content after second [SEP].

        Args:
            observation: Raw observation string

        Returns:
            str: Processed observation string
        """
        instruction_prefix = "Instruction: [SEP]"
        first_sep = observation.find(instruction_prefix)

        if first_sep == -1:
            return observation

        second_sep = observation.find(SEP_MARKER, first_sep + len(instruction_prefix))

        if second_sep == -1:
            return observation

        result = SEP_MARKER + observation[second_sep + len(SEP_MARKER):]
        return result

    def parse_instruction_in_envobs(
        self, instruction: str
    ) -> Tuple[str, str, List[str]]:
        """
        Parse instruction information in environment observation.

        Args:
            instruction: String containing instruction and page state

        Returns:
            Tuple[str, str, List[str]]: (observation, search_available, buttons)
                - observation: Page state
                - search_available: Whether search is available ("True" or "False")
                - buttons: List of clickable buttons
        """
        observation = instruction.split("搜索功能是否可用:")[0].strip()

        search_pattern = r"搜索功能是否可用: (True|False)"
        search_match = re.search(search_pattern, instruction)
        search_available = search_match.group(1) if search_match else "True"

        button_pattern = r"可点击的按钮:\s*\[(.*?)\]"
        button_match = re.search(button_pattern, instruction)
        if button_match:
            buttons_str = button_match.group(1)
            buttons = [btn.strip().strip('"') for btn in buttons_str.split(",") if btn.strip()]
        else:
            buttons = []

        return observation, search_available, buttons

    def act(
        self, user_role: str, instruction: str
    ) -> Tuple[bool, str, str]:
        """
        Execute one interaction step.

        Args:
            user_role: User role ("ENV" or "Shopper")
            instruction: Instruction content

        Returns:
            Tuple[bool, str, str]: (task_complete, role, observation)
                - task_complete: Whether task is complete
                - role: Next turn's role
                - observation: Observation result

        Raises:
            RuntimeError: When LLM call fails
            ValueError: When user_role is wrong or environment error occurs
        """
        if self.shopper_simulator is None or self.shop_env is None or self.tool_adapter is None:
            raise AttributeError("Agent not correctly connected to Shopper or ShopEnv, please check the configuration.")
        if user_role not in (ROLE_ENV, ROLE_SHOPPER):
            raise ValueError(f"user_role error: {user_role}")

        # Only the initial shopper input is a user message. Results from both
        # ShopEnv and ShopperSimulator are standard role=tool messages.
        if not self.messages or self.messages[-1].get("role") != "tool":
            prompt = f"{user_role}: {instruction}\n轮次: {self.turn}\n请做出下一轮决策:"
            self.messages.append({"role": "user", "content": prompt})

        if self.source not in ("idealab", "openai", "deepseek"):
            raise ValueError(f"Unsupported data source: {self.source}")
        assistant_message = self.get_response_idealab(self.messages)
        if assistant_message == FAILED_CALL_MESSAGE:
            raise RuntimeError("LLM call failed")

        self.messages.append(assistant_message)
        execution = self.tool_adapter.execute_assistant_message(assistant_message)
        self.messages.append(execution.message)

        if execution.name == "ask_shopper":
            return self.task_complete, ROLE_SHOPPER, execution.output["response"]

        self.action_list.append(execution.output["action"])
        if not execution.output.get("ok", False):
            raise ValueError(f"Environment error: {execution.output.get('error', 'unknown error')}")
        return self._handle_env_result(execution.output["result"])

    def _handle_env_result(self, env_response: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Handle a ShopEnv result already produced by the tool adapter."""
        observation = self.process_observation(env_response["instruction"])

        if env_response.get("done", False) or env_response.get("over", False):
            if self.shopper_simulator is None:
                raise AttributeError("shopper_simulator not initialized")
            self.save_to_json(
                env_response.get("reward", 0),
                env_response.get("reward_detail", {}),
                env_response.get("goal", {}),
                env_response.get("purchase", {}),
                self.shopper_simulator.messages,
            )
            self.task_complete = True

        return self.task_complete, ROLE_ENV, observation

    def _ask_shopper(self, question: str) -> str:
        """Execute the ask_shopper tool through the existing shopper simulator."""
        if self.shopper_simulator is None:
            raise AttributeError("shopper_simulator not initialized")
        return self.shopper_simulator.step(question)

    def _get_shopper_input(self) -> str:
        """
        Get latest reply from Shopper.

        Returns:
            str: Shopper's reply content

        Raises:
            AttributeError: When shopper_simulator is not initialized
        """
        if self.shopper_simulator is None:
            raise AttributeError("shopper_simulator not initialized")

        shopper_output = self.shopper_simulator.step("请提供您的购物需求。")
        if shopper_output:
            self.conversation_log.append({"shopper": shopper_output})
        return shopper_output

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

        for attempt in range(max_try):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tool_adapter.tools if self.tool_adapter else [],
                    tool_choice=self.config.get("tool_choice", "required"),
                    temperature=self.config.get("temperature", DEFAULT_TEMPERATURE),
                    max_tokens=self.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                )
                return assistant_message_to_dict(completion.choices[0].message)
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_try}): {e}")
                if attempt == max_try - 1:
                    traceback.print_exc()
                continue

        return FAILED_CALL_MESSAGE

    def save_to_json(
        self,
        reward: float,
        reward_detail: Dict[str, Any],
        goal: Dict[str, Any],
        purchase: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Save conversation and scores to JSON file.

        Args:
            reward: Reward score
            reward_detail: Reward details
            goal: Goal information
            purchase: Purchase information
            messages: Shopper conversation messages
        """
        log_data = {
            "task_id": self.task_id,
            "reward": reward,
            "reward_detail": reward_detail,
            "goal": goal,
            "purchase": purchase,
            "conversation": self.messages,
            "shopper_conversation": messages,
        }

        file_path = os.path.join(
            self.config["output_path"], self.config["model_name"]
        )
        os.makedirs(file_path, exist_ok=True)

        filename = os.path.join(file_path, f"{self.task_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        print(f"[LOG] Saved to file: {filename}")


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
                print(f"Warning: Unable to parse task ID '{task_id_str}', skipping file")

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
        shopper = ShopperSimulator(config["shopper_config"])
        shop_env = ShopEnv(config["env_config"])

        agent = Agent(task_id=task_id, config=config["agent_config"])

        agent.set_shopper(shopper)
        agent.set_shop_env(shop_env)
        task_description, observation = agent.reset()

        role = ROLE_SHOPPER
        agent.turn = 0
        agent.search_available = True
        agent.available_buttons = []
        agent.max_turns = config["agent_config"]["max_turns"]

        while True:
            done, role, observation = agent.act(role, observation)
            agent.turn += 1

            if agent.turn > agent.max_turns:
                if agent.shopper_simulator is not None:
                    agent.save_to_json(
                        0, {}, {}, {}, agent.shopper_simulator.messages
                    )
                if shop_env is not None:
                    shop_env.release()
                break

            if done:
                break

    except Exception as e:
        print(f"Error (task {task_id}): {e}")
        traceback.print_exc()
        if shop_env is not None:
            try:
                shop_env.release()
            except Exception:
                pass


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
        agent_config["output_path"], agent_config["model_name"]
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
