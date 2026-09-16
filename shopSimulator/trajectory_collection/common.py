"""Shared paths and serialization helpers for trajectory collection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "trajectory_collection/configs/collection.yaml"


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_collection_config(config: Dict[str, Any]) -> None:
    """Keep declared dataset sizes and their quotas in sync."""
    checks = (
        ("candidate_size", "difficulty_quotas"),
        ("online_dev_size", "online_dev_difficulty_quotas"),
    )
    for size_key, quota_key in checks:
        if int(config[size_key]) != sum(int(value) for value in config[quota_key].values()):
            raise ValueError(f"{size_key} must equal the sum of {quota_key}")

    accepted_total = sum(int(value) for value in config["accepted_difficulty_quotas"].values())
    split_total = sum(int(value) for value in config["sft_split_sizes"].values())
    if accepted_total != split_total:
        raise ValueError("accepted_difficulty_quotas must equal total sft_split_sizes")

    split_quotas = config["sft_split_difficulty_quotas"]
    for split_name, split_size in config["sft_split_sizes"].items():
        if int(split_size) != sum(int(value) for value in split_quotas[split_name].values()):
            raise ValueError(
                f"sft_split_sizes.{split_name} must equal its difficulty quota sum"
            )

    for difficulty, accepted in config["accepted_difficulty_quotas"].items():
        assigned = sum(int(split_quotas[name][difficulty]) for name in config["sft_split_sizes"])
        if int(accepted) != assigned:
            raise ValueError(
                f"accepted_difficulty_quotas.{difficulty} must equal its split quota sum"
            )


def load_collection_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Collection config must be a YAML mapping")
    validate_collection_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def collection_root(config: Dict[str, Any]) -> Path:
    return resolve_project_path(config["output_root"]) / config["collection_name"]


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Atomically replace a generated text artifact in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
