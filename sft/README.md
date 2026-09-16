# Qwen3-4B Action-only LoRA SFT

本项目使用 ShopSimulator 的购物轨迹，对 Qwen3-4B 进行单卡 BF16 LoRA
微调。模型只学习 assistant 的工具调用，不学习 system、user、环境返回或思考内容。

## 训练资源

- Conda 环境：`ft`
- 基础模型：`/root/autodl-tmp/Qwen3-4B`
- 清洗数据：`shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft_no_thought_prompt.jsonl`
- 正式配置：`configs/qwen3_4b_action_only_lora.json`
- 输出目录：`/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora`
- SwanLab 项目：`qwen3-shop-action-sft`

数据共有 1,486 条 trajectory。使用 seed 42 固定划分为 1,337 条训练数据和
149 条验证数据。基础模型及原始数据不会被覆盖。

## 启动正式训练

当前容器已安装 `screen`。推荐在 screen 会话中启动，避免 SSH 断开导致训练停止：

```bash
screen -S qwen3-sft
cd /root/ShopAgent-Training/sft
./scripts/run_train.sh
```

看到数据统计、模型加载信息和 SwanLab 链接后，说明训练已正常启动。

按 `Ctrl-a`，松开后再按 `d`，可以退出 screen 而不停止训练。重新进入训练窗口：

```bash
screen -r qwen3-sft
```

查看现有 screen 会话：

```bash
screen -ls
```

需要主动停止训练时，在训练窗口按 `Ctrl-c`。

## SwanLab

首次使用时登录：

```bash
conda run --no-capture-output -n ft swanlab login
```

检查当前登录状态：

```bash
conda run --no-capture-output -n ft swanlab verify
```

训练会上传 train/eval loss、learning rate、gradient norm、epoch、吞吐量、
模型配置、LoRA 配置和数据集大小。目前配置的项目地址为：

<https://swanlab.cn/@t1tt/qwen3-shop-action-sft>

## 训练参数

- 最大序列长度：8,192
- Epoch：3
- Micro batch size：1
- Gradient accumulation：8
- 有效 batch size：8
- Learning rate：`2e-4`
- Weight decay：`0.01`
- Warmup ratio：3%
- LoRA rank/alpha/dropout：`32 / 64 / 0.05`
- 每 5 个 update step 记录一次训练指标
- 每 25 个 update step 评估一次
- 每 50 个 update step 保存一次 checkpoint
- 保留全部 checkpoint，不自动删除
- 训练结束加载验证 loss 最低的 checkpoint

LoRA 应用于 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、
`up_proj` 和 `down_proj`。训练使用 BF16、SDPA、gradient checkpointing 和
fused AdamW。

## 训练前校验

只执行全量数据、chat template 和 loss mask 校验，不加载模型权重：

```bash
cd /root/ShopAgent-Training/sft
./scripts/run_train.sh --validate-only
```

运行真实 tokenizer 集成测试：

```bash
cd /root/ShopAgent-Training/sft
conda run --no-capture-output -n ft python tests/test_masking.py
```

查看某条数据实际学习和屏蔽的 token：

```bash
cd /root/ShopAgent-Training/sft
conda run --no-capture-output -n ft python scripts/inspect_mask.py --row 0
```

## 输出和恢复训练

正式训练输出位于：

```text
/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/
```

其中包括：

- `checkpoint-N/`：定期保存的 LoRA checkpoint
- `final_adapter/`：训练结束后的最佳 LoRA adapter 和 tokenizer
- `run_config.json`：本次实际配置
- `chat_template_non_thinking.jinja`：推理使用的非思考模板
- `swanlog/`：SwanLab 本地运行记录

从指定 checkpoint 恢复：

```bash
cd /root/ShopAgent-Training/sft
./scripts/run_train.sh \
  --resume-from-checkpoint /root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/checkpoint-N
```

把 `checkpoint-N` 替换为实际目录名。可以用下面的命令查看 checkpoint：

```bash
ls -d /root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/checkpoint-* 2>/dev/null
```

## 非思考模式和 loss mask

训练显式使用 `enable_thinking=False`。每个 assistant action 的实际格式为：

```text
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "search", "arguments": {"keywords": "..."}}
</tool_call><|im_end|>
```

assistant header 和空的 `<think>...</think>` 是推理模板预先提供的 token，标签为
`-100`。只有 `<tool_call>...</tool_call><|im_end|>` 参与 loss。以下内容全部被
屏蔽：

- system prompt 和工具定义
- user 消息
- tool/environment response
- assistant header
- 空 think 前缀
- padding
- 被截断的不完整 tool call

## 使用 vLLM 推理

推理必须使用训练输出中的非思考模板，避免训推格式不一致：

```bash
vllm serve /root/autodl-tmp/Qwen3-4B \
  --enable-lora \
  --lora-modules qwen3-action=/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/final_adapter \
  --chat-template /root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/chat_template_non_thinking.jinja \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

该模板默认关闭 thinking。调用方也可以显式传入：

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## 重新生成清洗数据

原始数据会被保留。需要重新生成清洗版本时运行：

```bash
cd /root/ShopAgent-Training/sft
conda run --no-capture-output -n ft python scripts/prepare_data.py \
  --input shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft.jsonl \
  --output shopSimulator/trajectory_collection/outputs/standard-single-deepseek-thinking-test50-v2/sft/accepted_sft_no_thought_prompt.jsonl
```
