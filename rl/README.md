# ShopSimulator GRPO (verl + LoRA)

This directory is isolated from `single_eval/` and contains the Qwen3-4B GRPO
LoRA path. It loads the already merged action-only SFT checkpoint as a normal
Hugging Face model and creates a new trainable GRPO LoRA on top of it.

## Reward V1

For a purchase, the compliance components are:

```text
Q = C_type * (0.45 * C_attr + 0.40 * C_option + 0.15 * C_price)
```

- `C_type` is a hard gate: exact ASIN or exact non-empty category path. Empty
  queries and broad category overlap never pass it.
- `C_attr` and `C_option` are matched fractions in `[0, 1]`.
- `C_price` uses the visible instruction budget and the selected SKU price.
- A fully compliant purchase gets `+1.0`.
- A wrong purchase gets `-1.0 + 0.2Q`, before small behavior penalties.
- Timeout/history/context limit gets `-1.05`; malformed/no-action termination
  gets `-1.10`.
- Buying an item worse than the best product actually shown incurs
  `-0.2 * max(0, Q_seen_best - Q_purchase)`.
- Same-state repeated actions cost `0.05` each, capped at `0.20`; invalid
  actions cost `0.05` each, capped at `0.10`.
- There is no step penalty and no direct reward for `back`. A useful back action
  is learned because it avoids the much larger irreversible wrong-buy penalty.
- Infrastructure failures raise `InfrastructureRewardError`, so they are not
  silently treated as model failures.

Reward diagnostics (`strict_success`, every compliance component, regret,
back/invalid/repeat/action counts) are returned to verl and logged separately.

## Data policy

The default train IDs are the existing 2,000 candidate IDs and validation uses
64 deterministic IDs from the disjoint online-dev manifest. Dataset generation
re-parses every visible budget and excludes tasks whose canonical requested SKU
cannot satisfy it. The exact product-data SHA256 and exclusions are recorded in
`/root/autodl-tmp/ShopAgent-Training/rl/data/manifest.json`.

Regenerate data:

```bash
cd /root/ShopAgent-Training
PYTHONPATH=$PWD:$PWD/shopSimulator /root/miniconda3/envs/rl/bin/python -m rl.prepare_dataset
```

## Run

Terminal 1 starts the existing ShopSimulator API with exactly 20 slots. Initial
loading can take several minutes:

```bash
cd /root/ShopAgent-Training
bash rl/start_shop_env.sh
```

Terminal 2 launches GRPO on every GPU visible to the `rl` environment (four
GPUs on the current host):

```bash
cd /root/ShopAgent-Training
bash rl/run_grpo.sh
```

Before a real run, validate the entire generate/tool/reward/update path with
one prompt, two rollouts, and one optimizer step:

```bash
bash rl/run_smoke.sh
```

The full launcher default is 4 prompts x 4 group samples = 16 concurrent
environment sessions. The launcher rejects values above 20. On the current
4-GPU host this gives each data-parallel worker four trajectories per update.
Common overrides:

```bash
TOTAL_EPOCHS=1 SAVE_FREQ=10 TEST_FREQ=10 \
GPU_MEMORY_UTILIZATION=0.45 bash rl/run_grpo.sh
```

To restrict a run to a subset of GPUs, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 N_GPUS_PER_NODE=2 bash rl/run_smoke.sh
```

Render and validate the complete Hydra configuration without training:

```bash
CONFIG_ONLY=1 bash rl/run_grpo.sh > /tmp/shopsim-grpo-config.txt
```

The launcher uses the already-installed `/root/miniconda3/envs/rl` environment,
auto-detects its visible GPU count, and trains a rank-32 LoRA with FSDP2 CPU
offload. It uses vLLM async multi-turn rollout, the non-thinking Qwen chat
template argument, a 42-assistant-turn limit, a 32,768-token model context, a
24,576-token response cap, and an explicit KL loss. `model.lora.merge=False` uses
verl's vLLM-native adapter synchronization: vLLM loads the merged SFT model as
the immutable base and receives only the newly trained GRPO LoRA weights. This
also avoids full-weight refits and preserves Qwen3's tied embedding/output
weight alias.
The default uses the padded SDPA path, so `flash-attn` is not required.
It never starts or stops a separate model server: verl owns rollout inference.

The launcher uses verl's synchronous trainer with its official async
multi-turn rollout path. It does not require the V1 TransferQueue trainer.
The ShopSimulator bridge delegates action validation and execution to the same
root `ShopToolAdapter` used by `single_eval`: invalid clicks and same-state
repeats are returned as recoverable observations, with the same limit of three
consecutive validation retries. Only orchestration is different: verl owns
generation and turn scheduling, while the bridge owns rollout-scoped
ShopSimulator allocation, cleanup, termination diagnostics, and reward traces.

Training also uses verl's native SwanLab logger. Set `SWANLAB_API_KEY` for
cloud sync, or use `SWANLAB_MODE=local` to keep a local run under
`/root/autodl-tmp/ShopAgent-Training/rl/outputs/.../swanlog`. Each step records reward means/std/min/max for score,
strict success, type/attribute/option/price compliance, quality, regret,
repeat/invalid/back/action counts, outcome and termination rates, plus the
within-group reward variance. Full per-trajectory values remain in the
`rollouts/*.jsonl` files.

The project pins the official verl Git revision and its tested core stack in
`requirements.txt`: vLLM 0.24, PyTorch 2.11, Transformers 5.9, and SwanLab
0.10. Do not mix
this revision with vLLM 0.29: that release renamed the LoRA mapper API and the
first in-memory adapter synchronization fails before rollout.

## Tests

```bash
cd /root/ShopAgent-Training
PYTHONPATH=$PWD:$PWD/shopSimulator /root/miniconda3/envs/rl/bin/python -m unittest discover -s rl/tests -v
```
