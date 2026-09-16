# ShopSimulator GRPO（verl + LoRA）

此目录与 `single_eval/` 相互独立，包含 Qwen3-4B 的 GRPO LoRA 训练流程。它会将已合并、仅包含动作数据的 SFT 检查点作为普通 Hugging Face 模型加载，并在其上创建一个新的、可训练的 GRPO LoRA。

## 奖励 V1

对于一次购买，合规性由以下部分组成：

```text
Q = C_type * (0.45 * C_attr + 0.40 * C_option + 0.15 * C_price)
```

- `C_type` 是一个硬门控条件：必须精确匹配 ASIN，或精确匹配非空的类目路径。空查询和宽泛的类目重叠均无法通过。
- `C_attr` 和 `C_option` 是 `[0, 1]` 范围内的匹配比例。
- `C_price` 使用指令中可见的预算和所选 SKU 的价格计算。
- 完全合规的购买获得 `+1.0` 奖励。
- 错误购买在扣除少量行为惩罚前，获得 `-1.0 + 0.2Q` 奖励。
- 超时、历史记录超限或上下文超限获得 `-1.05`；格式错误或无动作终止获得 `-1.10`。
- 如果购买的商品劣于实际展示过的最佳商品，则会受到 `-0.2 * max(0, Q_seen_best - Q_purchase)` 的惩罚。
- 在相同状态下重复执行动作，每次扣除 `0.05`，上限为 `0.20`；无效动作每次扣除 `0.05`，上限为 `0.10`。
- 不设步数惩罚，也不对 `back` 动作直接给予奖励。有效的返回动作之所以能被学会，是因为它可以避免代价大得多且不可逆的错误购买惩罚。
- 基础设施故障会抛出 `InfrastructureRewardError`，因此不会被静默地视为模型故障。

奖励诊断信息（`strict_success`、各项合规性指标、遗憾值，以及返回/无效/重复/动作计数）会返回给 verl，并分别记录日志。

## 数据策略

默认训练集 ID 为现有的 2,000 个候选 ID；验证集则使用来自互不重叠的 online-dev 清单中的 64 个确定性 ID。生成数据集时会重新解析每个可见预算，并排除其规范请求 SKU 无法满足预算的任务。产品数据的准确 SHA256 值和排除项记录在 `/root/autodl-tmp/ShopAgent-Training/rl/data/manifest.json` 中。

重新生成数据：

```bash
cd /root/ShopAgent-Training
PYTHONPATH=$PWD:$PWD/shopSimulator /root/miniconda3/envs/rl/bin/python -m rl.prepare_dataset
```

## 运行

终端 1 使用恰好 20 个槽位启动现有的 ShopSimulator API。首次加载可能需要几分钟：

```bash
cd /root/ShopAgent-Training
bash rl/start_shop_env.sh
```

终端 2 在 `rl` 环境可见的所有 GPU 上启动 GRPO（当前主机上为四张 GPU）：

```bash
cd /root/ShopAgent-Training
bash rl/run_grpo.sh
```

正式运行前，使用一个提示词、两次 rollout 和一个优化器步，对完整的生成/工具/奖励/更新流程进行验证：

```bash
bash rl/run_smoke.sh
```

完整启动器的默认配置为 4 个提示词 × 每组 4 个样本，即 16 个并发环境会话。启动器会拒绝大于 20 的值。在当前四 GPU 主机上，每个数据并行工作进程每次更新会处理四条轨迹。常用的覆盖配置如下：

```bash
TOTAL_EPOCHS=1 SAVE_FREQ=10 TEST_FREQ=10 \
GPU_MEMORY_UTILIZATION=0.45 bash rl/run_grpo.sh
```

如需将运行限制在部分 GPU 上，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 N_GPUS_PER_NODE=2 bash rl/run_smoke.sh
```

仅渲染并验证完整的 Hydra 配置，而不进行训练：

```bash
CONFIG_ONLY=1 bash rl/run_grpo.sh > /tmp/shopsim-grpo-config.txt
```

启动器使用已安装的 `/root/miniconda3/envs/rl` 环境，自动检测可见 GPU 数量，并通过 FSDP2 CPU 卸载训练秩为 32 的 LoRA。它使用 vLLM 异步多轮 rollout、Qwen 非思考模式的聊天模板参数、42 个助手轮次的上限、32,768 token 的模型上下文、24,576 token 的响应上限，以及显式 KL 损失。`model.lora.merge=False` 使用 verl 原生的 vLLM 适配器同步机制：vLLM 将已合并的 SFT 模型作为不可变的基座加载，并且只接收新训练的 GRPO LoRA 权重。这也避免了全量权重重新拟合，并保留 Qwen3 的嵌入层/输出层权重共享关系。

默认使用带填充的 SDPA 路径，因此不需要 `flash-attn`。
它不会启动或停止单独的模型服务器：rollout 推理由 verl 负责。

启动器使用 verl 的同步训练器及其官方异步多轮 rollout 流程，不需要 V1 TransferQueue 训练器。ShopSimulator 桥接层会将动作验证和执行委托给与 `single_eval` 相同的根级 `ShopToolAdapter`：无效点击和相同状态下的重复动作会作为可恢复的观察结果返回，并采用相同的连续三次验证重试限制。只有编排方式不同：verl 负责生成和轮次调度，而桥接层负责 rollout 范围内的 ShopSimulator 分配、清理、终止诊断和奖励轨迹。

训练还使用 verl 原生的 SwanLab 日志记录器。设置 `SWANLAB_API_KEY` 可同步到云端，或使用 `SWANLAB_MODE=local` 将本地运行记录保存在 `/root/autodl-tmp/ShopAgent-Training/rl/outputs/.../swanlog` 下。每一步都会记录以下指标的奖励均值/标准差/最小值/最大值：得分、严格成功率、类型/属性/选项/价格合规性、质量、遗憾值、重复/无效/返回/动作计数、结果率和终止率，以及组内奖励方差。每条轨迹的完整数值仍保存在 `rollouts/*.jsonl` 文件中。

项目在 `requirements.txt` 中固定了官方 verl Git 修订版本及其经过测试的核心技术栈：vLLM 0.24、PyTorch 2.11、Transformers 5.9 和 SwanLab 0.10。请勿将此修订版本与 vLLM 0.29 混用：该版本重命名了 LoRA mapper API，导致首次内存内适配器同步在 rollout 开始前失败。

## 测试

```bash
cd /root/ShopAgent-Training
PYTHONPATH=$PWD:$PWD/shopSimulator /root/miniconda3/envs/rl/bin/python -m unittest discover -s rl/tests -v
```
