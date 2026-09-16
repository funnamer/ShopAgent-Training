"""Small, isolated pipeline for collecting and curating teacher trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

import yaml

from trajectory_collection.common import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    atomic_text_writer,
    collection_root,
    load_collection_config,
    read_json,
    resolve_project_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from trajectory_collection.exporter import (
    export_reasoning_sample,
    export_sample,
    validate_exported_sample,
)
from trajectory_collection.sampler import (
    prepare_manifests,
    proportional_sample,
    stratified_sample,
)
from trajectory_collection.verifier import VerificationResult, load_and_verify

MAX_ENV_WORKERS = 20


def _manifest_path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return collection_root(dict(config)) / "manifests" / path


def _load_task_ids(path: Path) -> list[int]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Manifest must be a JSON list of integer task IDs: {path}")
    if len(value) != len(set(value)):
        raise ValueError(f"Manifest contains duplicate task IDs: {path}")
    return value


def _prepare_trial_manifest(config: Dict[str, Any], task_count: int) -> tuple[Path, Dict[str, int]]:
    prepared = prepare_manifests(config)
    manifest_path = collection_root(config) / "manifests" / f"trial{task_count}_ids.json"
    selected, quotas = proportional_sample(
        prepared["candidate_records"],
        task_count,
        config["difficulty_quotas"],
        random.Random(int(config["seed"]) + task_count),
    )
    expected_ids = [record["task_id"] for record in selected]
    if manifest_path.exists():
        if _load_task_ids(manifest_path) != expected_ids:
            raise ValueError(
                f"Existing trial manifest differs from the deterministic sample: {manifest_path}"
            )
    else:
        write_json(manifest_path, expected_ids)
    return manifest_path, quotas


def _load_teacher_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = resolve_project_path(config["teacher_config"])
    with path.open(encoding="utf-8") as handle:
        teacher = yaml.safe_load(handle)
    agent = teacher.get("agent_config") or {}
    env = teacher.get("env_config") or {}
    if agent.get("source") != "deepseek" or agent.get("thinking") != "enabled":
        raise ValueError("Teacher must be DeepSeek with thinking enabled")
    if env.get("if_persona", False):
        raise ValueError("Collection must use standard mode without persona")
    return teacher


def _import_single_eval_agent():
    single_eval = PROJECT_ROOT / "single_eval"
    if str(single_eval) not in sys.path:
        sys.path.insert(0, str(single_eval))
    import agent as single_agent

    return single_agent


def _task_metadata(config: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    path = collection_root(dict(config)) / "manifests" / "split_manifest.json"
    if not path.exists():
        return {}
    split = read_json(path)
    records = [
        *(split.get("candidate_records") or []),
        *(split.get("online_dev_records") or []),
        *(split.get("pilot_records") or []),
    ]
    return {int(record["task_id"]): record for record in records}


def _validate_manifest(
    config: Mapping[str, Any], task_ids: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    invalid = [
        task_id for task_id in task_ids
        if not int(config["train_start"]) <= task_id < int(config["train_end"])
    ]
    if invalid:
        raise ValueError(f"Manifest contains IDs outside train range: {invalid[:10]}")
    metadata = _task_metadata(config)
    missing = [task_id for task_id in task_ids if task_id not in metadata]
    if missing:
        raise ValueError(f"Manifest tasks are missing sampler metadata: {missing[:10]}")
    hashes = [metadata[task_id]["instruction_sha256"] for task_id in task_ids]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Manifest contains duplicate normalized instructions")
    return metadata


def _is_complete_pair(output_dir: Path, task_id: int) -> bool:
    trajectory_path = output_dir / f"{task_id}.json"
    diagnostics_path = output_dir / "diagnostics" / f"{task_id}.json"
    try:
        trajectory = read_json(trajectory_path)
        diagnostics = read_json(diagnostics_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(trajectory, dict)
        and isinstance(diagnostics, dict)
        and trajectory.get("task_id") == task_id
        and diagnostics.get("task_id") == task_id
    )


def collect(
    config: Dict[str, Any], manifest_path: Path, rollouts: int, max_workers: int
) -> Dict[str, Any]:
    task_ids = _load_task_ids(manifest_path)
    _validate_manifest(config, task_ids)
    if not 1 <= rollouts <= int(config["rollouts_per_task"]):
        raise ValueError("rollouts must be within the configured per-task cap")
    if not 1 <= max_workers <= MAX_ENV_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {MAX_ENV_WORKERS}")

    teacher = _load_teacher_config(config)
    single_agent = _import_single_eval_agent()
    single_agent.load_env_file()
    root = collection_root(config)
    prompt = teacher["agent_config"]["system_prompt"]
    summary = {
        "collection_name": config["collection_name"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "task_count": len(task_ids),
        "rollouts": rollouts,
        "max_workers": max_workers,
        "teacher_config_sha256": sha256_file(resolve_project_path(config["teacher_config"])),
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "teacher_model": teacher["agent_config"]["model_name"],
        "thinking": teacher["agent_config"]["thinking"],
        "runs": [],
    }
    for rollout_id in range(rollouts):
        run_name = f"rollout-{rollout_id:02d}"
        output_dir = root / "raw" / run_name
        finished = {task_id for task_id in task_ids if _is_complete_pair(output_dir, task_id)}
        todo = [task_id for task_id in task_ids if task_id not in finished]
        run_config = copy.deepcopy(teacher)
        run_config["agent_config"]["output_path"] = str(root / "raw")
        run_config["agent_config"]["run_name"] = run_name
        if todo:
            single_agent.run_tasks_multithreaded(todo, run_config, max_workers)
        complete = {task_id for task_id in task_ids if _is_complete_pair(output_dir, task_id)}
        write_json(output_dir / "run_metadata" / f"collection_{manifest_path.stem}.json", {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": summary["manifest_sha256"],
            "rollout_id": rollout_id,
            "task_ids": task_ids,
        })
        summary["runs"].append({
            "rollout_id": rollout_id,
            "completed_before_run": len(finished),
            "collected_now": len(todo),
            "completed": len(complete),
            "missing": sorted(set(task_ids) - complete),
        })
    write_json(root / "reports" / f"collection_status_{manifest_path.stem}.json", summary)
    return summary


def _quality_rank(result: VerificationResult, rollout_id: int) -> tuple:
    metrics = result.metrics
    return (
        metrics["repeated_action_count"],
        metrics["num_tool_calls"],
        metrics["completion_tokens"],
        rollout_id,
    )


def _attach_metadata(records: Sequence[Dict[str, Any]], metadata: Mapping[int, Dict[str, Any]]) -> None:
    keys = ("domain", "difficulty", "attribute_count", "option_count", "option_bucket", "instruction_sha256")
    for record in records:
        record.update({key: metadata[record["task_id"]][key] for key in keys})
        record.update({
            "teacher_thinking": True,
            "student_action_only": True,
            "prompt_label_conflict_acknowledged": True,
        })


def _record_rank(record: Mapping[str, Any]) -> tuple:
    metrics = record["verification"]["metrics"]
    return (
        metrics["repeated_action_count"], metrics["num_tool_calls"],
        metrics["completion_tokens"], record["rollout_id"], record["task_id"],
    )


def _apply_quotas(records: Sequence[Dict[str, Any]], quotas: Mapping[str, int]) -> tuple[list[Dict[str, Any]], Dict[str, int]]:
    accepted = []
    shortages = {}
    for difficulty in ("simple", "medium", "hard"):
        population = sorted((r for r in records if r["difficulty"] == difficulty), key=_record_rank)
        requested = int(quotas[difficulty])
        accepted.extend(population[:requested])
        shortages[difficulty] = max(0, requested - len(population))
    return sorted(accepted, key=lambda item: item["task_id"]), shortages


def _split_final_records(records: Sequence[Dict[str, Any]], config: Mapping[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    task_ids = [record["task_id"] for record in records]
    hashes = [record.get("instruction_sha256") for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Accepted records contain duplicate task IDs")
    if None in hashes or len(hashes) != len(set(hashes)):
        raise ValueError("Accepted records contain duplicate or missing instructions")
    remaining = list(records)
    result = {}
    rng = random.Random(int(config["seed"]) + 100)
    for name in ("train", "validation", "reserve"):
        chosen = stratified_sample(remaining, config["sft_split_difficulty_quotas"][name], rng)
        result[name] = chosen
        chosen_ids = {row["task_id"] for row in chosen}
        remaining = [row for row in remaining if row["task_id"] not in chosen_ids]
    if remaining or sum(map(len, result.values())) != len(records):
        raise ValueError("Final split did not assign every accepted record exactly once")
    return result


def _export(
    records: Sequence[Dict[str, Any]], *, include_reasoning: bool = False
) -> list[Dict[str, Any]]:
    samples = []
    for record in records:
        with Path(record["trajectory_path"]).open(encoding="utf-8") as handle:
            trajectory = json.load(handle)
        sample = (
            export_reasoning_sample(trajectory)
            if include_reasoning
            else export_sample(trajectory)
        )
        validate_exported_sample(sample, allow_reasoning=include_reasoning)
        samples.append(sample)
    return samples


def _write_pilot_report(root: Path, report: Mapping[str, Any]) -> None:
    reasons = "\n".join(f"| `{key}` | {value} |" for key, value in report["rejection_counts"].items()) or "| — | 0 |"
    tasks = "\n".join(
        f"| {task_id} | {count}/4 | {'accepted' if count else 'rejected'} |"
        for task_id, count in report["task_success_counts"].items()
    )
    text = "\n".join([
        "# Pilot quality report", "",
        f"- Tasks: {report['task_count']}",
        f"- Raw rollouts: {report['expected_rollouts']}",
        f"- Hard-pass rollouts: {report['hard_pass_rollout_count']} ({report['rollout_hard_pass_rate']:.1%})",
        f"- Accepted tasks: {report['accepted_task_count']} ({report['task_acceptance_rate']:.1%})",
        f"- Mean selected tool calls: {report['mean_tool_calls']:.3f}", "",
        "## Per-task hard-pass count", "", "| Task | Valid rollouts | Result |", "|---:|---:|---|", tasks, "",
        "## Rejection reasons", "", "One rollout may have multiple reasons.", "", "| Reason | Count |", "|---|---:|", reasons, "",
    ])
    with atomic_text_writer(root / "reports" / "pilot_report.md") as handle:
        handle.write(text)


def curate(config: Dict[str, Any], manifest_path: Path, rollouts: int) -> Dict[str, Any]:
    if not 1 <= rollouts <= int(config["rollouts_per_task"]):
        raise ValueError("rollouts must be within the configured per-task cap")
    task_ids = _load_task_ids(manifest_path)
    metadata = _validate_manifest(config, task_ids)
    teacher = _load_teacher_config(config)
    prompt_hash = hashlib.sha256(teacher["agent_config"]["system_prompt"].encode()).hexdigest()
    root = collection_root(config)
    rejected, selected, observed = [], [], []
    success_counts = {}

    for task_id in task_ids:
        valid = []
        for rollout_id in range(rollouts):
            run_dir = root / "raw" / f"rollout-{rollout_id:02d}"
            trajectory = run_dir / f"{task_id}.json"
            diagnostics = run_dir / "diagnostics" / f"{task_id}.json"
            if not trajectory.exists():
                rejected.append({"task_id": task_id, "rollout_id": rollout_id, "reasons": ["missing_trajectory"]})
                continue
            try:
                verification = load_and_verify(
                    trajectory, diagnostics, int(config["train_start"]), int(config["train_end"]),
                    expected_task_id=task_id,
                    expected_system_prompt_sha256=prompt_hash,
                    expected_instruction_sha256=metadata[task_id]["instruction_sha256"],
                    expected_model_name=teacher["agent_config"]["model_name"],
                    expected_run_name=f"rollout-{rollout_id:02d}",
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                rejected.append({"task_id": task_id, "rollout_id": rollout_id, "reasons": ["unreadable_trajectory"], "error": str(error)})
                continue
            observed.append(verification.metrics)
            record = {
                "task_id": task_id, "rollout_id": rollout_id,
                "trajectory_path": str(trajectory.resolve()),
                "diagnostics_path": str(diagnostics.resolve()),
                "verification": verification.to_dict(),
            }
            if verification.accepted:
                valid.append((verification, rollout_id, record))
            else:
                rejected.append({"task_id": task_id, "rollout_id": rollout_id, "reasons": verification.reasons, "metrics": verification.metrics})
        success_counts[task_id] = len(valid)
        if valid:
            selected.append(min(valid, key=lambda row: _quality_rank(row[0], row[1]))[2])

    _attach_metadata(selected, metadata)
    shortages = {difficulty: 0 for difficulty in ("simple", "medium", "hard")}
    if manifest_path.name == "candidate_ids.json":
        selected, shortages = _apply_quotas(selected, config["accepted_difficulty_quotas"])

    rejection_counts = Counter(reason for row in rejected for reason in row["reasons"])
    selected_metrics = [row["verification"]["metrics"] for row in selected]
    hard_passes = sum(success_counts.values())
    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "teacher_config_sha256": sha256_file(resolve_project_path(config["teacher_config"])),
        "system_prompt_sha256": prompt_hash,
        "task_count": len(task_ids),
        "expected_rollouts": len(task_ids) * rollouts,
        "accepted_task_count": len(selected),
        "hard_pass_rollout_count": hard_passes,
        "rollout_hard_pass_rate": hard_passes / (len(task_ids) * rollouts) if task_ids else 0,
        "task_acceptance_rate": len(selected) / len(task_ids) if task_ids else 0,
        "task_success_counts": {str(key): value for key, value in sorted(success_counts.items())},
        "task_success_histogram": dict(sorted(Counter(success_counts.values()).items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "acceptance_quota_shortages": shortages,
        "mean_tool_calls": sum(m["num_tool_calls"] for m in selected_metrics) / len(selected_metrics) if selected_metrics else 0,
        "mean_repeated_actions": sum(m["repeated_action_count"] for m in selected_metrics) / len(selected_metrics) if selected_metrics else 0,
        "all_rollout_token_usage": {
            "prompt_tokens": sum(m["prompt_tokens"] for m in observed),
            "completion_tokens": sum(m["completion_tokens"] for m in observed),
        },
        "exact_asin_rate": sum(m["exact_asin_match"] for m in selected_metrics) / len(selected_metrics) if selected_metrics else 0,
        "exact_options_rate": sum(m["exact_options_match"] for m in selected_metrics) / len(selected_metrics) if selected_metrics else 0,
        "accepted_distribution": {
            key: dict(sorted(Counter(str(row.get(key, "unknown")) for row in selected).items()))
            for key in ("difficulty", "domain", "option_bucket")
        },
    }
    write_json(root / "reports" / f"rejection_report_{manifest_path.stem}.json", rejected)
    selected_path = root / "sft" / f"selected_manifest_{manifest_path.stem}.jsonl"
    write_jsonl(selected_path, selected)
    report["selected_manifest_sha256"] = sha256_file(selected_path)

    if manifest_path.name == "pilot_ids.json":
        output_name = "pilot_sft.jsonl"
    elif manifest_path.name == "candidate_ids.json":
        output_name = "accepted_sft.jsonl"
    else:
        output_name = f"{manifest_path.stem}_sft.jsonl"
    write_jsonl(root / "sft" / output_name, _export(selected))
    reasoning_output_name = output_name.replace("_sft.jsonl", "_reasoning_sft.jsonl")
    write_jsonl(
        root / "sft" / reasoning_output_name,
        _export(selected, include_reasoning=True),
    )
    if manifest_path.name == "pilot_ids.json":
        _write_pilot_report(root, report)

    if manifest_path.name == "candidate_ids.json":
        split_paths = {
            name: root / "sft" / f"sft_{name}.jsonl"
            for name in config["sft_split_sizes"]
        }
        reasoning_split_paths = {
            name: root / "sft" / f"sft_{name}_reasoning.jsonl"
            for name in config["sft_split_sizes"]
        }
        expected = sum(int(v) for v in config["sft_split_sizes"].values())
        if len(selected) == expected:
            for name, records in _split_final_records(selected, config).items():
                write_jsonl(split_paths[name], _export(records))
                write_jsonl(
                    reasoning_split_paths[name],
                    _export(records, include_reasoning=True),
                )
            report["sft_splits_written"] = True
        else:
            for path in (*split_paths.values(), *reasoning_split_paths.values()):
                path.unlink(missing_ok=True)
            report["sft_splits_written"] = False

    write_json(root / "reports" / f"quality_report_{manifest_path.stem}.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def local_config(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", default=argparse.SUPPRESS)

    prepare = subparsers.add_parser("prepare", help="Create fixed manifests once")
    local_config(prepare)
    collect_parser = subparsers.add_parser("collect", help="Collect or resume rollouts")
    local_config(collect_parser)
    collect_parser.add_argument("--manifest", required=True)
    collect_parser.add_argument("--rollouts", type=int)
    collect_parser.add_argument("--max-workers", type=int)
    curate_parser = subparsers.add_parser("curate", help="Verify, select and export")
    local_config(curate_parser)
    curate_parser.add_argument("--manifest", required=True)
    curate_parser.add_argument("--rollouts", type=int)
    trial_parser = subparsers.add_parser(
        "trial", help="Prepare, collect and curate a proportional trial sample"
    )
    local_config(trial_parser)
    trial_parser.add_argument("--tasks", type=int, required=True)
    trial_parser.add_argument("--rollouts", type=int)
    trial_parser.add_argument("--max-workers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_collection_config(args.config)
    if args.command == "prepare":
        prepared = prepare_manifests(config)
        result = {key: prepared[key] for key in ("seed", "audit", "deduplication", "candidate", "online_dev", "pilot")}
    elif args.command == "collect":
        result = collect(config, _manifest_path(config, args.manifest), args.rollouts or int(config["rollouts_per_task"]), args.max_workers or int(config["max_workers"]))
    elif args.command == "curate":
        result = curate(config, _manifest_path(config, args.manifest), args.rollouts or int(config["rollouts_per_task"]))
    else:
        rollouts = args.rollouts or int(config["rollouts_per_task"])
        max_workers = args.max_workers or int(config["max_workers"])
        manifest_path, quotas = _prepare_trial_manifest(config, args.tasks)
        collection = collect(config, manifest_path, rollouts, max_workers)
        missing = sorted({task_id for run in collection["runs"] for task_id in run["missing"]})
        result = {
            "manifest": str(manifest_path.resolve()),
            "task_count": args.tasks,
            "difficulty_quotas": quotas,
            "collection": collection,
            "curation": curate(config, manifest_path, rollouts),
        }
        if missing:
            result["missing"] = missing
            result["message"] = (
                "Missing rollouts were rejected; curation selected the best of the "
                "remaining valid rollouts. Rerun the same command only if you want "
                "to backfill the missing raw trajectories."
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
