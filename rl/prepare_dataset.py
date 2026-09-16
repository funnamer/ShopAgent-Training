"""Create verl parquet datasets and audit impossible budget tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset

from rl.budget import extract_price_upper


ROOT = Path(__file__).resolve().parents[1]
SHOP_SIMULATOR_ROOT = ROOT / "shopSimulator"
STORAGE_ROOT = Path("/root/autodl-tmp/ShopAgent-Training/rl")
DEFAULT_PRODUCTS = SHOP_SIMULATOR_ROOT / "shop_env/data/fine_items_eval_train_all.json"
DEFAULT_TRAIN_IDS = SHOP_SIMULATOR_ROOT / "trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/manifests/candidate_ids.json"
DEFAULT_VAL_IDS = SHOP_SIMULATOR_ROOT / "trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/manifests/online_dev_ids.json"
DEFAULT_OUTPUT = STORAGE_ROOT / "data"


def _norm(value: Any) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", str(value or "")).casefold() if ch.isalnum())


def _target_option_prices(item: dict[str, Any], goal_options: list[str]) -> list[float]:
    prices: list[float] = []
    for wanted in goal_options:
        wanted_norm = _norm(wanted)
        matches = []
        for values in (item.get("customization_options") or {}).values():
            for option in values or []:
                value_norm = _norm(option.get("value"))
                if wanted_norm and value_norm == wanted_norm and option.get("price") is not None:
                    matches.append(float(option["price"]))
        if matches:
            prices.append(min(matches))
    if not prices:
        prices = [float(x) for x in item.get("pricing") or []]
    return prices


def _goal(task_id: int, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    instruction = item["instructions"][0]
    text = instruction["instruction"]
    upper = extract_price_upper(text)
    goal_options = list(instruction.get("instruction_options") or instruction.get("options") or [])
    prices = _target_option_prices(item, goal_options)
    goal = {
        "asin": str(item["asin"]),
        "category": item.get("category") or "",
        "name": item.get("title") or "",
        "attributes": list(instruction.get("attributes") or []),
        "goal_options": goal_options,
        "price_upper": upper,
        "instruction_text": text,
    }
    audit = {
        "task_id": task_id,
        "price_upper": upper,
        "target_option_prices": prices,
        "budget_conflict": bool(upper is not None and prices and min(prices) > upper + 1e-8),
    }
    return goal, audit


def _row(task_id: int, goal: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    initial = (
        f"Instruction: {goal['instruction_text']}"
        "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
    )
    tool_kwargs = {
        name: {"create_kwargs": {"task_id": task_id}}
        for name in ("search", "click")
    }
    return {
        "data_source": "shopsimulator_grpo",
        "agent_name": "shop_tool_agent",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial},
        ],
        "ability": "shopping",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"task_id": task_id, "goal": goal}, ensure_ascii=False),
        },
        "extra_info": {
            "index": task_id,
            "task_id": task_id,
            "need_tools_kwargs": True,
            "tool_selection": ["search", "click"],
            "tools_kwargs": tool_kwargs,
        },
    }


def _load_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as handle:
        return [int(x) for x in json.load(handle)]


def build_split(
    ids: list[int], products: list[dict[str, Any]], system_prompt: str, limit: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    if limit > 0:
        ids = ids[:limit]
    rows, audit = [], []
    for task_id in ids:
        goal, record = _goal(task_id, products[task_id])
        audit.append(record)
        if not record["budget_conflict"]:
            rows.append(_row(task_id, goal, system_prompt))
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN_IDS)
    parser.add_argument("--val-ids", type=Path, default=DEFAULT_VAL_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.products.open(encoding="utf-8") as handle:
        products = json.load(handle)
    system_prompt = (ROOT / "rl/configs/system_prompt.txt").read_text(encoding="utf-8").strip()
    train_rows, train_audit = build_split(
        _load_ids(args.train_ids), products, system_prompt, args.train_limit, args.seed
    )
    val_rows, val_audit = build_split(
        _load_ids(args.val_ids), products, system_prompt, args.val_limit, args.seed + 1
    )
    if not train_rows or not val_rows:
        raise RuntimeError("dataset is empty after budget-conflict filtering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(str(args.output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(args.output_dir / "val.parquet"))
    product_hash = hashlib.sha256(args.products.read_bytes()).hexdigest()
    manifest = {
        "seed": args.seed,
        "products": str(args.products.resolve()),
        "products_sha256": product_hash,
        "train": {
            "requested": len(train_audit),
            "kept": len(train_rows),
            "excluded_budget_conflict_ids": [x["task_id"] for x in train_audit if x["budget_conflict"]],
        },
        "validation": {
            "requested": len(val_audit),
            "kept": len(val_rows),
            "excluded_budget_conflict_ids": [x["task_id"] for x in val_audit if x["budget_conflict"]],
        },
        "reward_budget_policy": "visible instruction; approximate values receive 10% tolerance",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
