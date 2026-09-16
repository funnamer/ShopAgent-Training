"""Fail-fast checks before a costly GRPO launch."""

from __future__ import annotations

import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import requests
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path("/root/autodl-tmp/ShopAgent-Training/rl")
SUPPORTED_VERSIONS = {
    "verl": SpecifierSet(">=0.9,<0.11", prereleases=True),
    "torch": SpecifierSet("==2.11.*"),
    "vllm": SpecifierSet("==0.24.*"),
    "transformers": SpecifierSet("==5.9.*"),
    "swanlab": SpecifierSet("==0.10.*"),
}


def _check_versions() -> dict[str, str]:
    installed = {name: version(name) for name in SUPPORTED_VERSIONS}
    unsupported = [
        f"{name}=={installed[name]} (expected {constraint})"
        for name, constraint in SUPPORTED_VERSIONS.items()
        if installed[name] not in constraint
    ]
    if unsupported:
        raise SystemExit("Unsupported GRPO package versions:\n" + "\n".join(unsupported))
    return installed


def _check_qwen3_lora_bridge() -> None:
    """Require verl's official vLLM>=0.25 packed-LoRA name fix."""
    from verl.utils.vllm import VLLMHijack

    source = inspect.getsource(VLLMHijack.hijack)
    if "get_unstacked_mapper" not in source:
        raise SystemExit(
            "Installed verl lacks the official vLLM>=0.25 Qwen3 LoRA mapper fix.\n"
            "Install the verl revision pinned in rl/requirements.txt before running."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/Qwen3-4B"))
    parser.add_argument("--train", type=Path, default=STORAGE_ROOT / "data/train.parquet")
    parser.add_argument("--val", type=Path, default=STORAGE_ROOT / "data/val.parquet")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoTokenizer

    required = [
        args.model / "config.json",
        args.train,
        args.val,
        ROOT / "rl/configs/tools.yaml",
        ROOT / "rl/configs/agent_loop.yaml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not list(args.model.glob("*.safetensors")):
        missing.append(f"{args.model}/*.safetensors")
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))

    versions = _check_versions()
    _check_qwen3_lora_bridge()
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if model_config.model_type != "qwen3":
        raise SystemExit(f"Expected a Qwen3 model, got model_type={model_config.model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        raise SystemExit(f"Model tokenizer has no chat template: {args.model}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    response = requests.post(
        f"{args.base_url.rstrip('/')}/api/shop_agent",
        json={"action": "release_all"},
        timeout=5,
    )
    response.raise_for_status()
    print(
        json.dumps(
            {
                "status": "ok",
                **versions,
                "torch": torch.__version__,
                "model_type": model_config.model_type,
                "gpu": torch.cuda.get_device_name(0),
                "shop_env": response.json(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
