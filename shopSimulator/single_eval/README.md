# Single-turn evaluation

本目录用于运行 ShopSimulator 单轮购物评测。下面记录本地 Qwen3-4B 基座模型与
LoRA checkpoint 的启动、评测和结果统计流程。

## 1. 配置独立的 checkpoint 评测

为 checkpoint 使用单独的 `run_name`，可以让结果继续保存在
`outputs/standard/` 下，同时避免覆盖基座模型结果。

首次评测 checkpoint-150 时，可以复制基座模型配置：

```bash
cd /root/ShopAgent-Training/shopSimulator/single_eval
cp configs/standard/qwen3_4b_local.yaml \
   configs/standard/qwen3_4b_checkpoint_150.yaml
```

在 `configs/standard/qwen3_4b_checkpoint_150.yaml` 中设置：

```yaml
agent_config:
  # 该名称必须与 serve_vllm.py 暴露给客户端的模型名称一致。
  model_name: qwen3-4b-action-only
  run_name: qwen3-4b-action-only-checkpoint-150
  base_url: http://127.0.0.1:8000/v1
  task_nums: 300
  output_path: outputs/standard
```

比较不同模型时，两份配置的 `task_nums` 和任务选择配置必须保持一致。只修改
`run_name`，不要把 `model_name` 改成 checkpoint 名称。

对应的输出目录是：

```text
outputs/standard/qwen3-4b-action-only-local/
outputs/standard/qwen3-4b-action-only-checkpoint-150/
```

## 2. 启动购物环境

在终端 1 中运行：

```bash
cd /root/ShopAgent-Training/shopSimulator/shop_env/shop_env
python pack_api.py
```

环境默认监听 `http://127.0.0.1:5000`。

必须从内层 `shop_env/shop_env` 目录启动。若从外层目录直接执行
`python shop_env/pack_api.py`，可能因为相对模块路径不正确而出现：

```text
ModuleNotFoundError: No module named 'web_agent_site'
```

启动时出现 Gym 0.24 的版本提示只是警告，不是上述导入失败的原因。

## 3. 启动模型服务

### 加载 checkpoint-150

在终端 2 中运行：

```bash
cd /root/ShopAgent-Training/shopSimulator/single_eval
python serve_vllm.py checkpoint-150
```

默认 checkpoint 路径为：

```text
/root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/checkpoint-150
```

也可以传入 checkpoint 的绝对路径：

```bash
python serve_vllm.py \
  /root/autodl-tmp/sft_outputs/qwen3-4b-action-only-lora/checkpoint-150
```

模型服务默认监听 `http://127.0.0.1:8000/v1`。启动后可检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

客户端看到的模型名称应为 `qwen3-4b-action-only`。

### 加载未微调基座模型

需要重新测试基座模型时，先停止占用 8000 端口的 checkpoint 服务，然后运行：

```bash
cd /root/ShopAgent-Training/shopSimulator/single_eval
python serve_vllm.py
```

## 4. 单线程运行评测

确认购物环境和模型服务均已启动，然后在终端 3 中运行 checkpoint-150 评测：

```bash
cd /root/ShopAgent-Training/shopSimulator/single_eval
python agent.py \
  --yaml_name configs/standard/qwen3_4b_checkpoint_150.yaml
```

命令中不添加 `--multithread`，因此会按单线程顺序执行任务。

基座模型评测命令为：

```bash
python agent.py --yaml_name configs/standard/qwen3_4b_local.yaml
```

评测支持断点续跑：重新运行同一条命令时，已有顶层 `<task_id>.json` 的任务会被
跳过，没有生成顶层结果文件的任务会被重试。

## 5. 按官方规则统计结果

`get_score.py` 只读取指定输出目录顶层的 JSON，不读取 `diagnostics/`。从仓库根
目录统计 checkpoint-150：

```bash
cd /root/ShopAgent-Training/shopSimulator
python - <<'PY'
from get_score import get_finished_task, calculate_metrics, print_metrics

path = "single_eval/outputs/standard/qwen3-4b-action-only-checkpoint-150"
print_metrics(calculate_metrics(get_finished_task(path)))
PY
```

统计基座模型时，将 `path` 改为：

```text
single_eval/outputs/standard/qwen3-4b-action-only-local
```

官方指标的具体规则如下：

- `reward_detail` 非空才计为完成。
- `r_hard` 是 `r_type * r_att * r_option * r_price`。
- 四个子分数都等于 1 才计为完全成功。
- `purchase.asin == goal.asin` 才计为选对商品。
- `reward_detail` 为空时，完成、子分数、hard、success 和选对商品均按 0 计。

## 6. 比较结果时处理未保存任务

如果模型调用报错或达到限制后没有生成顶层结果文件，官方 `get_score.py` 会完全
排除该任务，样本总量和分母也会随之减少。这会造成不同模型之间的结果覆盖不一致，
并可能高估失败较多的模型。

比较时先检查两个目录的顶层文件数量：

```bash
cd /root/ShopAgent-Training/shopSimulator
find single_eval/outputs/standard/qwen3-4b-action-only-local \
  -maxdepth 1 -type f -name '*.json' | wc -l
find single_eval/outputs/standard/qwen3-4b-action-only-checkpoint-150 \
  -maxdepth 1 -type f -name '*.json' | wc -l
```

推荐采用以下两种口径之一：

1. 重新运行失败任务，直到两个模型都生成相同任务集合的完整结果。
2. 固定原始任务总数，将没有保存结果的任务作为失败并按 0 分计入分母。

只比较双方共同保存的任务也可以进行配对分析，但会排除发生调用错误或未保存结果的
任务，因此不能反映完整的端到端完成能力。
