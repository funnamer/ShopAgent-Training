"""Deterministic, train-only stratified task sampling."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from trajectory_collection.common import (
    collection_root,
    read_json,
    resolve_project_path,
    write_json,
)


DIFFICULTIES = ("simple", "medium", "hard")


def normalize_instruction(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def difficulty_for(attribute_count: int) -> str:
    if attribute_count <= 3:
        return "simple"
    if attribute_count <= 6:
        return "medium"
    return "hard"


def _task_record(task_id: int, item: Mapping[str, Any]) -> Dict[str, Any]:
    instruction_data = item["instructions"][0]
    instruction = str(instruction_data["instruction"])
    attribute_count = len(instruction_data.get("attributes") or [])
    option_count = len(instruction_data.get("instruction_options") or [])
    return {
        "task_id": task_id,
        "domain": str(item.get("domain_zh") or "未知类目"),
        "difficulty": difficulty_for(attribute_count),
        "attribute_count": attribute_count,
        "option_count": option_count,
        "option_bucket": "multi" if option_count >= 2 else "single",
        "instruction_sha256": hashlib.sha256(
            normalize_instruction(instruction).encode("utf-8")
        ).hexdigest(),
    }


def load_eligible_tasks(
    data_file: Path, train_start: int, train_end: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with data_file.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("Product data must be a JSON array")
    if train_end > len(items):
        raise ValueError(f"train_end={train_end} exceeds dataset size {len(items)}")

    eval_instructions = {
        normalize_instruction(str(item["instructions"][0]["instruction"]))
        for item in items[:train_start]
    }
    eligible: List[Dict[str, Any]] = []
    excluded_eval_overlap: List[int] = []
    invalid_tag: List[int] = []
    for task_id in range(train_start, train_end):
        item = items[task_id]
        if item.get("tag") != "train":
            invalid_tag.append(task_id)
            continue
        instruction = normalize_instruction(
            str(item["instructions"][0]["instruction"])
        )
        if instruction in eval_instructions:
            excluded_eval_overlap.append(task_id)
            continue
        eligible.append(_task_record(task_id, item))

    audit = {
        "dataset_size": len(items),
        "train_range": [train_start, train_end],
        "eligible_count": len(eligible),
        "excluded_eval_instruction_overlap_count": len(excluded_eval_overlap),
        "excluded_eval_instruction_overlap_ids": excluded_eval_overlap,
        "invalid_train_tag_ids": invalid_tag,
    }
    return eligible, audit


def _allocate_largest_remainder(
    groups: Mapping[Tuple[str, str], Sequence[Dict[str, Any]]], total: int
) -> Dict[Tuple[str, str], int]:
    nonempty = {key: values for key, values in groups.items() if values}
    capacity = sum(len(values) for values in nonempty.values())
    if total > capacity:
        raise ValueError(f"Requested {total} samples from capacity {capacity}")
    if total < len(nonempty):
        raise ValueError("Quota is too small to represent every non-empty stratum")

    ideals = {
        key: total * len(values) / capacity for key, values in nonempty.items()
    }
    allocation = {
        key: min(len(nonempty[key]), max(1, math.floor(ideal)))
        for key, ideal in ideals.items()
    }

    while sum(allocation.values()) > total:
        removable = [key for key, value in allocation.items() if value > 1]
        if not removable:
            raise ValueError("Unable to reduce stratum allocation to requested total")
        key = min(removable, key=lambda item: (ideals[item] - allocation[item], item))
        allocation[key] -= 1

    while sum(allocation.values()) < total:
        expandable = [
            key
            for key, value in allocation.items()
            if value < len(nonempty[key])
        ]
        if not expandable:
            raise ValueError("Unable to fill stratum allocation to requested total")
        key = max(expandable, key=lambda item: (ideals[item] - allocation[item], item))
        allocation[key] += 1
    return allocation


def stratified_sample(
    records: Sequence[Dict[str, Any]],
    quotas: Mapping[str, int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    by_difficulty: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_difficulty[record["difficulty"]].append(record)

    for difficulty in DIFFICULTIES:
        requested = int(quotas[difficulty])
        if requested <= 0:
            continue
        strata: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for record in by_difficulty[difficulty]:
            strata[(record["domain"], record["option_bucket"])].append(record)
        allocation = _allocate_largest_remainder(strata, requested)
        for key in sorted(allocation):
            population = sorted(strata[key], key=lambda item: item["task_id"])
            selected.extend(rng.sample(population, allocation[key]))

    return sorted(selected, key=lambda item: item["task_id"])


def proportional_sample(
    records: Sequence[Dict[str, Any]],
    total: int,
    weights: Mapping[str, int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Sample an exact total using the configured difficulty proportions."""
    if total < 1:
        raise ValueError("Trial task count must be positive")
    if total > len(records):
        raise ValueError(f"Requested {total} trial tasks from {len(records)} candidates")

    weight_total = sum(int(weights[difficulty]) for difficulty in DIFFICULTIES)
    if weight_total <= 0:
        raise ValueError("Difficulty weights must have a positive sum")
    ideals = {
        difficulty: total * int(weights[difficulty]) / weight_total
        for difficulty in DIFFICULTIES
    }
    quotas = {difficulty: math.floor(ideals[difficulty]) for difficulty in DIFFICULTIES}
    remaining = total - sum(quotas.values())
    order = sorted(
        DIFFICULTIES,
        key=lambda difficulty: (ideals[difficulty] - quotas[difficulty], difficulty),
        reverse=True,
    )
    for difficulty in order[:remaining]:
        quotas[difficulty] += 1

    selected: List[Dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        population = sorted(
            (record for record in records if record["difficulty"] == difficulty),
            key=lambda record: record["task_id"],
        )
        if quotas[difficulty] > len(population):
            raise ValueError(
                f"Not enough {difficulty} candidates for quota {quotas[difficulty]}"
            )
        selected.extend(rng.sample(population, quotas[difficulty]))
    return sorted(selected, key=lambda record: record["task_id"]), quotas


def deduplicate_and_refill(
    selected: Sequence[Dict[str, Any]], eligible_pool: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove normalized-instruction duplicates and refill the exact stratum."""
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    used_hashes = set()
    selected_ids = {record["task_id"] for record in selected}
    for record in sorted(selected, key=lambda item: item["task_id"]):
        instruction_hash = record["instruction_sha256"]
        if instruction_hash in used_hashes:
            removed.append(record)
        else:
            kept.append(record)
            used_hashes.add(instruction_hash)

    replacements = []
    available = sorted(
        (
            record
            for record in eligible_pool
            if record["task_id"] not in selected_ids
            and record["instruction_sha256"] not in used_hashes
        ),
        key=lambda item: item["task_id"],
    )
    for duplicate in removed:
        stratum = (
            duplicate["difficulty"],
            duplicate["domain"],
            duplicate["option_bucket"],
        )
        replacement = next(
            (
                record
                for record in available
                if (
                    record["difficulty"],
                    record["domain"],
                    record["option_bucket"],
                )
                == stratum
                and record["instruction_sha256"] not in used_hashes
            ),
            None,
        )
        if replacement is None:
            raise ValueError(f"Unable to refill duplicate instruction in stratum {stratum}")
        kept.append(replacement)
        replacements.append(replacement)
        used_hashes.add(replacement["instruction_sha256"])
        available = [
            record for record in available if record["task_id"] != replacement["task_id"]
        ]

    result = sorted(kept, key=lambda item: item["task_id"])
    return result, {
        "removed_duplicate_task_ids": [record["task_id"] for record in removed],
        "replacement_task_ids": [record["task_id"] for record in replacements],
        "remaining_internal_duplicate_count": len(result)
        - len({record["instruction_sha256"] for record in result}),
    }


def choose_pilot(
    candidates: Sequence[Dict[str, Any]], pilot_size: int, seed: int
) -> List[Dict[str, Any]]:
    by_domain: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_domain[record["domain"]].append(record)
    domains = sorted(by_domain)
    if pilot_size < len(domains):
        raise ValueError("pilot_size must cover every domain")

    rng = random.Random(seed)
    selected = [rng.choice(by_domain[domain]) for domain in domains]
    selected_ids = {record["task_id"] for record in selected}

    largest_domain = max(domains, key=lambda domain: (len(by_domain[domain]), domain))
    priority_pool = [
        record
        for record in by_domain[largest_domain]
        if record["difficulty"] == "hard"
        and record["option_bucket"] == "multi"
        and record["task_id"] not in selected_ids
    ]
    remaining_pool = [
        record for record in candidates if record["task_id"] not in selected_ids
    ]
    while len(selected) < pilot_size:
        pool = priority_pool if priority_pool else remaining_pool
        choice = rng.choice(pool)
        selected.append(choice)
        selected_ids.add(choice["task_id"])
        priority_pool = [r for r in priority_pool if r["task_id"] != choice["task_id"]]
        remaining_pool = [r for r in remaining_pool if r["task_id"] != choice["task_id"]]
    return sorted(selected, key=lambda item: item["task_id"])


def summarize(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    return {
        "count": len(records),
        "difficulty": dict(sorted(Counter(r["difficulty"] for r in records).items())),
        "domain": dict(sorted(Counter(r["domain"] for r in records).items())),
        "option_bucket": dict(
            sorted(Counter(r["option_bucket"] for r in records).items())
        ),
    }


def prepare_manifests(config: Dict[str, Any]) -> Dict[str, Any]:
    manifest_dir = collection_root(config) / "manifests"
    required = [
        manifest_dir / "candidate_ids.json",
        manifest_dir / "online_dev_ids.json",
        manifest_dir / "pilot_ids.json",
        manifest_dir / "split_manifest.json",
    ]
    existing = [path.exists() for path in required]
    if all(existing):
        return read_json(manifest_dir / "split_manifest.json")
    if any(existing):
        raise ValueError(
            "Manifest set is incomplete. Restore the missing files or use a new "
            "collection_name; prepare will not silently resample IDs."
        )

    data_file = resolve_project_path(config["data_file"])
    eligible, audit = load_eligible_tasks(
        data_file, int(config["train_start"]), int(config["train_end"])
    )
    rng = random.Random(int(config["seed"]))
    candidates = stratified_sample(eligible, config["difficulty_quotas"], rng)
    candidates, candidate_deduplication = deduplicate_and_refill(candidates, eligible)
    candidate_ids = {record["task_id"] for record in candidates}
    candidate_hashes = {record["instruction_sha256"] for record in candidates}
    remaining = [
        record
        for record in eligible
        if record["task_id"] not in candidate_ids
        and record["instruction_sha256"] not in candidate_hashes
    ]
    online_dev = stratified_sample(
        remaining, config["online_dev_difficulty_quotas"], rng
    )
    online_dev, online_dev_deduplication = deduplicate_and_refill(
        online_dev, remaining
    )
    pilot = choose_pilot(
        candidates, int(config["pilot_size"]), int(config["seed"]) + 1
    )

    write_json(manifest_dir / "candidate_ids.json", [r["task_id"] for r in candidates])
    write_json(manifest_dir / "online_dev_ids.json", [r["task_id"] for r in online_dev])
    write_json(manifest_dir / "pilot_ids.json", [r["task_id"] for r in pilot])
    result = {
        "seed": config["seed"],
        "audit": audit,
        "deduplication": {
            "candidate": candidate_deduplication,
            "online_dev": online_dev_deduplication,
            "candidate_online_dev_instruction_overlap_count": len(
                candidate_hashes
                & {record["instruction_sha256"] for record in online_dev}
            ),
        },
        "candidate": summarize(candidates),
        "online_dev": summarize(online_dev),
        "pilot": summarize(pilot),
        "candidate_records": candidates,
        "online_dev_records": online_dev,
        "pilot_records": pilot,
    }
    write_json(manifest_dir / "split_manifest.json", result)
    return result
