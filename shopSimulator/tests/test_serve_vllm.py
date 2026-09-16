import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "single_eval" / "serve_vllm.py"
SPEC = importlib.util.spec_from_file_location("serve_vllm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
serve_vllm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_vllm)


def make_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_model = tmp_path / "base"
    checkpoints_root = tmp_path / "outputs"
    vllm_bin = tmp_path / "vllm"
    base_model.mkdir()
    checkpoints_root.mkdir()
    (base_model / "config.json").write_text("{}", encoding="utf-8")
    (checkpoints_root / "chat_template_non_thinking.jinja").write_text(
        "template", encoding="utf-8"
    )
    vllm_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    vllm_bin.chmod(0o755)
    return base_model, checkpoints_root, vllm_bin


def parse_for_layout(
    base_model: Path,
    checkpoints_root: Path,
    vllm_bin: Path,
    checkpoint: str | None = None,
):
    argv = [] if checkpoint is None else [checkpoint]
    argv.extend(
        [
            "--base-model",
            str(base_model),
            "--checkpoints-root",
            str(checkpoints_root),
            "--vllm-bin",
            str(vllm_bin),
        ]
    )
    return serve_vllm.parse_args(argv)


class ServeVllmTest(unittest.TestCase):
    def test_no_checkpoint_serves_original_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model, checkpoints_root, vllm_bin = make_layout(Path(temp_dir))
            args = parse_for_layout(base_model, checkpoints_root, vllm_bin)

            command, checkpoint = serve_vllm.build_command(args)

            self.assertIsNone(checkpoint)
            self.assertNotIn("--enable-lora", command)
            self.assertEqual(
                command[command.index("--served-model-name") + 1], args.model_name
            )

    def test_checkpoint_enables_lora_with_stable_api_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model, checkpoints_root, vllm_bin = make_layout(Path(temp_dir))
            checkpoint = checkpoints_root / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "'
                + str(base_model)
                + '", "r": 32}',
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            args = parse_for_layout(
                base_model, checkpoints_root, vllm_bin, "checkpoint-100"
            )

            command, resolved_checkpoint = serve_vllm.build_command(args)

            self.assertEqual(resolved_checkpoint, checkpoint.resolve())
            self.assertIn("--enable-lora", command)
            self.assertIn("--max-lora-rank", command)
            self.assertIn(f"{args.model_name}={checkpoint.resolve()}", command)
            self.assertEqual(
                command[command.index("--served-model-name") + 1],
                "qwen3-4b-base",
            )

    def test_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model, checkpoints_root, vllm_bin = make_layout(Path(temp_dir))
            args = parse_for_layout(
                base_model, checkpoints_root, vllm_bin, "checkpoint-999"
            )

            with self.assertRaisesRegex(ValueError, "Invalid LoRA checkpoint"):
                serve_vllm.build_command(args)


if __name__ == "__main__":
    unittest.main()
