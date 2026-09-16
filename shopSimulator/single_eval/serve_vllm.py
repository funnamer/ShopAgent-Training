"""Serve the base Qwen3 model or one of its LoRA checkpoints with vLLM.

Examples:
    python serve_vllm.py
    python serve_vllm.py checkpoint-100
    python serve_vllm.py /path/to/checkpoint-100
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_CHECKPOINTS_ROOT = Path(
    "/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora"
)
DEFAULT_VLLM_BIN = Path("/root/miniconda3/envs/ft/bin/vllm")
DEFAULT_MODEL_NAME = "qwen3-4b-action-only"
BASE_MODEL_NAME = "qwen3-4b-base"
DEFAULT_MAX_MODEL_LEN = 32768


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start an OpenAI-compatible vLLM server. With no checkpoint, serve "
            "the original model; with checkpoint-xxx, apply that LoRA adapter."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help=(
            "LoRA checkpoint name under --checkpoints-root (for example "
            "checkpoint-100), or an absolute checkpoint path. Omit it to use "
            "the original model."
        ),
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name used by OpenAI API clients in both base and LoRA modes.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the vLLM command without starting it.",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(value: str | None, checkpoints_root: Path) -> Path | None:
    if value is None:
        return None

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = checkpoints_root / candidate
    return candidate.resolve()


def validate_paths(
    base_model: Path,
    checkpoints_root: Path,
    checkpoint: Path | None,
    vllm_bin: Path,
) -> Path:
    base_model = base_model.expanduser().resolve()
    checkpoints_root = checkpoints_root.expanduser().resolve()
    vllm_bin = vllm_bin.expanduser().resolve()

    if not (base_model / "config.json").is_file():
        raise ValueError(f"Invalid base model directory: {base_model}")
    if not vllm_bin.is_file() or not os.access(vllm_bin, os.X_OK):
        raise ValueError(f"vLLM executable is missing or not executable: {vllm_bin}")

    template = checkpoints_root / "chat_template_non_thinking.jinja"
    if not template.is_file():
        raise ValueError(f"Non-thinking chat template not found: {template}")

    if checkpoint is not None:
        adapter_config_path = checkpoint / "adapter_config.json"
        adapter_model_path = checkpoint / "adapter_model.safetensors"
        if not adapter_config_path.is_file() or not adapter_model_path.is_file():
            raise ValueError(
                "Invalid LoRA checkpoint; expected adapter_config.json and "
                f"adapter_model.safetensors in {checkpoint}"
            )

        with adapter_config_path.open(encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        configured_base = adapter_config.get("base_model_name_or_path")
        if configured_base and Path(configured_base).resolve() != base_model:
            raise ValueError(
                f"Checkpoint expects base model {configured_base}, not {base_model}"
            )
        rank = adapter_config.get("r")
        if rank != 32:
            raise ValueError(
                f"This launcher expects LoRA rank 32, but {checkpoint} has r={rank}"
            )

    return template.resolve()


def build_command(args: argparse.Namespace) -> tuple[list[str], Path | None]:
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoints_root)
    template = validate_paths(
        args.base_model,
        args.checkpoints_root,
        checkpoint,
        args.vllm_bin,
    )
    base_model = args.base_model.expanduser().resolve()
    vllm_bin = args.vllm_bin.expanduser().resolve()

    command = [
        str(vllm_bin),
        "serve",
        str(base_model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--chat-template",
        str(template),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    if checkpoint is None:
        command.extend(["--served-model-name", args.model_name])
    else:
        # Keep the client-facing adapter name stable so the same eval YAML works
        # for every checkpoint. Give the underlying base model a different name
        # to avoid a collision in vLLM's model registry.
        command.extend(
            [
                "--served-model-name",
                BASE_MODEL_NAME,
                "--enable-lora",
                "--max-lora-rank",
                "32",
                "--lora-modules",
                f"{args.model_name}={checkpoint}",
            ]
        )

    return command, checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        command, checkpoint = build_command(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    mode = f"LoRA ({checkpoint})" if checkpoint else "original base model"
    print(f"Serving mode: {mode}")
    print(f"OpenAI base URL: http://127.0.0.1:{args.port}/v1")
    print(f"API model name: {args.model_name}")
    print("Command:", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
