# ShopSimulator teacher trajectory collection

这是一个独立于 `single_eval` 输出的轻量采集目录。它直接复用现有 Agent、工具和
ShopEnv，不修改环境、reward、商品数据或原 eval 的默认行为。

固定约定：standard single-turn、无 persona、原 system prompt、DeepSeek thinking
教师、每题 4 次、Best-of-4，以及面向 Qwen3-4B 的 action-only tool-call 导出。

## 使用命令

快速试跑可以用一条命令完成固定清单生成、按难度比例抽样、采集和整理：

```bash
python trajectory_collection/pipeline.py trial --tasks 50
```

默认使用配置中的 `rollouts_per_task` 和 `max_workers`；当前即每题 4 个 rollout、20 个
worker。难度数量按 `difficulty_quotas` 的比例自动分配，当前 50 题会得到 15 simple、
25 medium、10 hard。任务清单固定保存为 `manifests/trial50_ids.json`，命令中断后直接
重跑即可断点续采。缺失 rollout 会按无效轨迹记录，curate 仍会从同一任务剩余的有效
rollout 中选择最佳结果；需要补齐原始轨迹时再重跑同一命令。

```bash
# 1. 固定 seed=42 创建清单。完整清单已存在时只读取，不重新采样。
python trajectory_collection/pipeline.py prepare \
  --config trajectory_collection/configs/collection.yaml

# 2. 采集或断点续跑。完整性的唯一单位是 (task_id, rollout_id)，
#    且原始轨迹和 diagnostics 两个 JSON 都必须存在且 task_id 一致。
python trajectory_collection/pipeline.py collect \
  --config trajectory_collection/configs/collection.yaml \
  --manifest pilot_ids.json \
  --rollouts 4 \
  --max-workers 8

# 3. 硬过滤、Best-of-4、报告和 action-only 导出。
python trajectory_collection/pipeline.py curate \
  --config trajectory_collection/configs/collection.yaml \
  --manifest pilot_ids.json \
  --rollouts 4
```

确认 pilot 后，把两处 `pilot_ids.json` 换成 `candidate_ids.json` 即可批量采集和
整理。程序不会在采集或整理时重新生成 ID，也不使用文件锁、清单版本状态机或扩容
审批流程。每次采集和整理报告都会记录输入 manifest 的 SHA256。

## 输出

```text
trajectory_collection/outputs/<collection_name>/
├── raw/rollout-00..03/       # thinking 教师的完整原始轨迹和 diagnostics
├── manifests/                # 固定 ID 清单与分层元数据
├── reports/                  # 按 manifest 命名的完整度、拒绝原因和质量报告
└── sft/                      # action-only JSONL 与按 manifest 命名的审计 sidecar
```

`pilot_sft.jsonl` / `accepted_sft.jsonl` 是 action-only 版本：教师
`reasoning_content` 被删除，assistant `content` 为空，`arguments` 为 JSON object。
对应的 `*_reasoning_sft.jsonl` 保留教师原始 `content` 和 `reasoning_content`，方便后续
按训练任务自行保留、删除或转换 thinking。两种版本都会清空最后一条包含 reward 和目标
信息的 tool response；未经处理的完整 thinking 始终保存在 `raw/` 中。

`selected_manifest_<manifest>.jsonl`、collection/quality/rejection report 均按输入
manifest 隔离。只有整理 `candidate_ids.json` 才会生成最终 train/validation/reserve；
数量不足时会移除同目录中可能残留的旧 split，避免误用。

训练时仍应使用 Qwen3 原生 chat template，并设置 `enable_thinking=False` 和
`assistant_only_loss=True`。
