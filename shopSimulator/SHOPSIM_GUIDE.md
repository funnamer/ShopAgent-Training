# ShopSimulator 启动与配置指南

本文档说明如何启动 ShopSimulator 文本环境，运行单轮/多轮评测，以及采集 DeepSeek 教师轨迹。

所有命令默认从仓库根目录执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
```

## 1. 快速启动

已经安装依赖、准备商品数据并配置 `.env` 时，使用两个终端即可启动。

终端 A：启动购物环境服务并保持进程运行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/shop_env/shop_env
python pack_api.py
```

终端 B：运行最小的单轮 Smoke Test：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/single_eval
python agent.py --yaml_name configs/standard/deepseek_smoke.yaml
```

如果要采集教师轨迹，终端 B 改为：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator

python trajectory_collection/pipeline.py trial --tasks 50
```

`trial` 会自动执行 prepare、按配置难度比例抽取指定数量、collect 和 curate。当前配置的
难度比例为 30% simple、50% medium、20% hard，因此 `--tasks 50` 会固定抽取
15 simple、25 medium、10 hard；默认每题 4 个 rollout、8 个 worker。中途中断或有任务
失败时，缺失 rollout 会作为无效轨迹，程序仍会从该任务其余有效 rollout 中执行
Best-of-the-rest 并导出结果。若希望补齐全部原始轨迹，重新运行同一条命令即可继续。
需要临时覆盖默认值时可追加
`--rollouts 4 --max-workers 8`。

教师轨迹采集的主配置位于：

```text
ShopSimulator/trajectory_collection/configs/collection.yaml
```

其中 `teacher_config` 指向实际使用的教师模型配置：

```text
ShopSimulator/single_eval/configs/standard/deepseek_thinking.yaml
```

各命令的职责：

| 命令 | 作用 |
| --- | --- |
| `prepare` | 按固定 seed 生成 candidate、online-dev 和 pilot 任务清单；已有完整清单时不会重新抽样 |
| `collect` | 按 manifest 调用教师模型采集原始轨迹，支持按 `(task_id, rollout_id)` 断点续跑 |
| `curate` | 硬校验轨迹、执行 Best-of-N、生成质量报告，并同时导出 action-only 与保留 reasoning 的 SFT JSONL |
| `trial --tasks N` | 一键执行上述完整流程，只需指定总任务数 N；难度数量按 `difficulty_quotas` 比例自动计算 |

`collection.yaml` 中最常修改的参数：

| 参数 | 含义 |
| --- | --- |
| `collection_name` | 本次采集名称，同时也是 `trajectory_collection/outputs/` 下的目录名 |
| `teacher_config` | DeepSeek 教师模型 YAML 路径；模型名、thinking、system prompt 和 token 上限在该文件中配置 |
| `data_file` | 用于抽样和核对任务的商品数据文件 |
| `output_root` | manifest、原始轨迹、报告和 SFT 文件的输出根目录 |
| `seed` | 任务抽样随机种子；相同数据与配置会得到相同任务清单 |
| `train_start` / `train_end` | 允许采集的 train 任务 ID 范围，左闭右开 |
| `candidate_size` | 正式候选任务数，必须等于 `difficulty_quotas` 三档数量之和 |
| `online_dev_size` | 独立在线开发集任务数，必须等于 `online_dev_difficulty_quotas` 之和 |
| `pilot_size` | pilot 清单包含的任务数 |
| `rollouts_per_task` | 每个任务最多采集多少次，用于 Best-of-N |
| `max_workers` | 默认并发 Agent 数，不能大于环境槽位数 `SHOPSIM_ENV_MAX_NUM` |
| `accepted_difficulty_quotas` | 正式整理后 simple/medium/hard 各保留多少条 |
| `sft_split_sizes` | 最终 train/validation/reserve 的样本数 |
| `sft_split_difficulty_quotas` | 每个最终 split 内部的难度配额 |

采集命令行参数会覆盖本次运行行为，但不会修改 YAML：

| 参数 | 含义 |
| --- | --- |
| `--config` | 指定主采集配置文件 |
| `--manifest` | 指定要运行的任务清单；只写文件名时从当前 collection 的 `manifests/` 中读取 |
| `--rollouts` | 本次使用的 rollout 数，范围为 `1..rollouts_per_task` |
| `--max-workers` | 本次实际并发数；建议与 `SHOPSIM_ENV_MAX_NUM` 保持一致 |

例如 `--manifest pilot_ids.json --rollouts 4` 表示对 pilot 中每个任务最多采集四次，
然后由 `curate` 从每个任务通过硬校验的轨迹中选择质量最好的一条。修改规模或配额后，
建议同时更换 `collection_name`，避免新旧采集产物混在同一目录。

并发采集时，环境槽位数应不小于 `--max-workers`。可在启动终端 A 前临时设置：

```bash
export SHOPSIM_ENV_MAX_NUM=8
```

检查环境服务是否可用：

```bash
curl -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"release_all"}'
```

如果提示端口 5000 已被占用，先运行 `lsof -nP -iTCP:5000 -sTCP:LISTEN`。若监听进程是
`python pack_api.py`，说明环境服务已经启动，无需再次启动。

首次使用还需要完成依赖、数据和 `.env` 配置，请继续阅读后面的对应章节。

## 2. 项目结构与运行关系

```text
LongHorizonAgentRL/
├── AGENTS.md                         # 项目修改约束；YAML 中的提示词禁止修改
├── .env                              # 本地 API Key、模型地址和环境并发数
├── agent_rl/                         # 后续 RL 训练与实验代码目录
└── ShopSimulator/
    ├── SHOPSIM_GUIDE.md              # 当前中文启动与配置指南
    ├── README.md                     # ShopSimulator 项目原始说明
    ├── requirements.txt              # 聚合项目依赖
    ├── tool_adapter.py               # OpenAI tool call 到环境动作的公共适配层
    ├── task_selection.py             # 可复现任务抽样与选择逻辑
    ├── get_score.py                  # 评测结果统计入口
    ├── shop_env/
    │   ├── requirements.txt          # 环境服务依赖
    │   ├── data/
    │   │   ├── fine_items_eval_train_all.json     # 完整商品和任务数据
    │   │   └── items_eval_train.json              # 环境默认读取入口
    │   ├── shop_env/
    │   │   ├── pack_api.py           # Flask API 服务，默认监听 5000 端口
    │   │   ├── shop_agent.py         # 环境会话、动作执行和资源管理
    │   │   └── shop_agent.log        # 环境运行日志
    │   ├── search_engine/
    │   │   └── products.sqlite3      # 自动构建的 SQLite FTS5 搜索索引
    │   ├── web_agent_site/
    │   │   ├── envs/                 # Gym 文本购物环境和状态机
    │   │   └── engine/               # 搜索、商品、目标和 reward 逻辑
    │   └── tests/                     # 搜索与环境测试
    ├── single_eval/
    │   ├── agent.py                  # 单轮购物 Agent 和批量执行入口
    │   ├── env.py                    # 对 Flask ShopEnv API 的客户端封装
    │   ├── configs/
    │   │   ├── standard/             # 完整用户要求模式配置
    │   │   └── persona/              # 带用户画像模式配置
    │   ├── scripts/                   # 常用单轮评测脚本
    │   └── outputs/                   # 单轮评测生成的轨迹和 diagnostics
    ├── multi_eval/
    │   ├── agent.py                  # 多轮购物 Agent
    │   ├── shopper.py                # 模拟用户及澄清对话逻辑
    │   ├── env.py                    # 多轮环境客户端
    │   ├── configs/                  # standard/persona 配置
    │   ├── scripts/                  # 常用多轮评测脚本
    │   └── outputs/                   # 多轮评测结果
    ├── trajectory_collection/
    │   ├── pipeline.py               # prepare/collect/curate 三阶段入口
    │   ├── sampler.py                # train-only 分层抽样和去重
    │   ├── verifier.py               # reward、动作、泄漏等硬校验
    │   ├── exporter.py               # action-only SFT JSONL 导出
    │   ├── common.py                 # 配置、路径和原子写入工具
    │   ├── configs/collection.yaml   # 教师采集规模与配额
    │   ├── tests/                     # 采集流水线单元测试
    │   └── outputs/<collection>/
    │       ├── manifests/             # 固定任务 ID 和分层元数据
    │       ├── raw/                   # 各 rollout 原始 thinking 轨迹
    │       ├── reports/               # 完整度、拒绝原因和质量报告
    │       └── sft/                   # action-only 训练数据与审计 sidecar
    └── tests/                         # 公共工具和任务选择测试
```

`outputs/`、日志和搜索索引都是运行产物；核心代码、配置与测试位于它们的同级目录。
所有 YAML 配置中的提示词均视为用户维护的只读内容；可以按任务调整采集规模等非提示词
参数，但不得修改 `system_prompt`、`prompt` 或其他模型指令文本。

一次完整评测或轨迹采集的调用关系如下：

```text
DeepSeek API
    ↑
single_eval/agent.py 或 trajectory_collection/pipeline.py
    ↓ 调用模型并生成标准 tool call
tool_adapter.py             校验并转换 search/click
    ↓ HTTP :5000
shop_env/shop_env/pack_api.py
    ↓
WebAgentTextEnv             执行搜索、点击、选择规格和购买
    ↓
SQLite FTS5 + Original Reward
```

主要边界如下：

- `single_eval`、`multi_eval` 和 `trajectory_collection` 都通过 HTTP 调用同一个环境服务；
- `tool_adapter.py` 只转换动作格式，不修改环境状态和 reward；
- `trajectory_collection` 复用 `single_eval` 的 Agent，但使用独立输出目录，不覆盖评测结果；
- 商品数据与 reward 由 `shop_env` 管理，模型 API Key 仅由客户端 Agent 使用。

## 3. Conda 环境

当前使用的环境名称是 `shopsim`：

```bash
conda activate shopsim
python --version
```

推荐 Python 3.10。

安装购物环境依赖：

```bash
pip install -r ShopSimulator/shop_env/requirements.txt
```

安装评测客户端额外需要的依赖：

```bash
pip install openai PyYAML requests
```

安装原始 Reward 使用的中文 spaCy 模型：

```bash
pip install \
  https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
```

验证关键依赖：

```bash
python -c "import gym, flask, spacy, openai, yaml; spacy.load('zh_core_web_sm'); print('dependencies OK')"
```

## 4. 商品数据

环境代码默认读取：

```text
ShopSimulator/shop_env/data/items_eval_train.json
```

仓库提供的是压缩文件：

```text
ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz
```

如果尚未解压，执行：

```bash
cd ShopSimulator/shop_env/data
gzip -dk fine_items_eval_train_all.json.gz
ln -sfn fine_items_eval_train_all.json items_eval_train.json
cd ../../..
```

执行后应存在：

```text
data/fine_items_eval_train_all.json
data/items_eval_train.json -> fine_items_eval_train_all.json
```

## 5. `.env` 配置

仓库根目录使用 `.env` 保存本地密钥和运行参数：

```dotenv
DEEPSEEK_API_KEY=替换为你的APIKey
DEEPSEEK_BASE_URL=https://api.deepseek.com
SHOPSIM_ENV_MAX_NUM=1
```

`.env` 已被 `.gitignore` 忽略，不应提交到版本库。建议设置权限：

```bash
chmod 600 .env
```

字段说明：

| 字段 | 含义 | 推荐值 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 从 DeepSeek 控制台获取 |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容接口地址 | `https://api.deepseek.com` |
| `SHOPSIM_ENV_MAX_NUM` | 同时初始化的环境槽位数 | 本地测试用 `1` |

`single_eval/agent.py` 和 `pack_api.py` 都会通过 `python-dotenv` 自动读取仓库根目录的 `.env`。如果终端中已经显式设置了同名环境变量，终端中的值优先。

## 6. 启动购物环境

在终端 A 执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/shop_env/shop_env
python pack_api.py
```

服务默认监听：

```text
http://127.0.0.1:5000/api/shop_agent
```

首次启动时，新的 SQLite FTS5 搜索引擎会自动构建索引：

```text
ShopSimulator/shop_env/search_engine/products.sqlite3
```

首次构建需要读取完整商品数据，因此会比后续启动慢。商品文件发生变化时，索引会自动重建。

可以在另一个终端检查服务：

```bash
curl -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"release_all"}'
```

正常响应：

```json
{
  "result": {
    "message": "All environments have been initialized"
  }
}
```

停止服务时，在终端 A 按 `Ctrl+C`。

## 7. 运行 DeepSeek Smoke Test

确认购物环境正在运行后，在终端 B 执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/single_eval

python agent.py \
  --yaml_name configs/standard/deepseek_smoke.yaml
```

`agent.py` 会自动读取根目录 `.env`，因此这里不需要再次执行 `source .env`。

Smoke 配置通过顺序抽样执行 task 0、1、2：

```yaml
task_nums: 1495
task_selection:
  mode: sequential
  sample_size: 3
  seed: 42
output_path: outputs/test_tool3
```

结果保存在：

```text
ShopSimulator/single_eval/outputs/test_tool3/deepseek-flash/0.json
```

如果结果文件已经存在，断点续跑逻辑会认为 task 0 已完成。需要重新测试时，请先将该 JSON 移到备份位置，然后重新执行命令。

## 8. DeepSeek 评测配置说明

当前配置文件：

```text
ShopSimulator/single_eval/configs/standard/deepseek_smoke.yaml
```

完整结构：

```yaml
env_config:
  base_url: http://127.0.0.1:5000

agent_config:
  model_name: deepseek-flash
  source: deepseek
  model_key_env: DEEPSEEK_API_KEY
  base_url_env: DEEPSEEK_BASE_URL
  thinking: disabled
  max_tokens: 512
  system_prompt: |
    你正在进行网上购物……
  task_nums: 1495
  task_selection:
    mode: sequential
    sample_size: 3
    seed: 42
  output_path: outputs/test_tool3
```

### `env_config`

| 字段 | 含义 |
| --- | --- |
| `base_url` | ShopSimulator HTTP 服务地址；代码会追加 `/api/shop_agent` |
| `if_persona` | 可选；设为 `true` 时启用 Persona 模式 |

Standard 模式不设置 `if_persona`：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
```

Persona 模式设置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
  if_persona: true
```

### `agent_config`

| 字段 | 含义 |
| --- | --- |
| `model_name` | 传给模型服务的模型名称 |
| `source` | API 类型；当前支持 `deepseek`、`openai`、`idealab` |
| `model_key_env` | 从哪个环境变量读取 API Key |
| `base_url_env` | 从哪个环境变量读取模型服务 URL |
| `thinking` | DeepSeek 思考模式；评测建议使用 `disabled` |
| `max_tokens` | 每一步模型最大输出 token 数 |
| `temperature` | 可选；默认 `0.0`，保证评测更稳定 |
| `tool_choice` | 可选；默认 `required`，要求模型每轮调用一个工具 |
| `system_prompt` | 购物 Agent 的动作规范和任务提示 |
| `task_nums` | 候选任务池大小，对应任务 ID `[0, task_nums)` |
| `task_selection.mode` | 可选；`sequential`（默认）或 `random` |
| `task_selection.sample_size` | 可选；从候选池选择多少个任务，默认选择全部 |
| `task_selection.seed` | 随机模式使用的整数 seed，默认 `0` |
| `output_path` | 结果输出根目录，相对于 `single_eval` 当前目录 |

配置中只保存环境变量名，不直接保存 API Key：

```yaml
model_key_env: DEEPSEEK_API_KEY
base_url_env: DEEPSEEK_BASE_URL
```

## 9. 运行更多任务

建议复制 smoke 配置：

```bash
cd ShopSimulator/single_eval
cp configs/standard/deepseek_smoke.yaml \
   configs/standard/deepseek_eval.yaml
```

然后修改。下面表示从 1495 个候选任务中，用 seed 42 可复现地随机抽取 100 个：

```yaml
task_nums: 1495
task_selection:
  mode: random
  sample_size: 100
  seed: 42
output_path: outputs/standard
```

如果需要按任务 ID 顺序运行前 100 个任务：

```yaml
task_nums: 1495
task_selection:
  mode: sequential
  sample_size: 100
  seed: 42  # sequential 模式会忽略 seed
```

相同的 `task_nums`、`sample_size` 和 `seed` 总会得到相同的随机任务 ID。
实际选择结果还会保存到：

```text
<output_path>/<model_name>/run_metadata/task_selection.json
```

断点续跑时会先按相同配置重新生成完整抽样，再跳过其中已经完成的任务，
因此不会因为部分任务已经完成而改变剩余样本。`sequential` 控制任务 ID 的
选择顺序；使用 `--multithread` 时完成顺序仍由各任务耗时决定，严格串行请不要
传 `--multithread`。

单线程运行：

```bash
python agent.py \
  --yaml_name configs/standard/deepseek_eval.yaml
```

多线程运行：

```bash
python agent.py \
  --yaml_name configs/standard/deepseek_eval.yaml \
  --multithread \
  --max_workers 4
```

并发时需要保证环境槽位数不少于 Agent worker 数。例如：

```dotenv
SHOPSIM_ENV_MAX_NUM=4
```

修改 `.env` 后需要重启购物环境服务。

## 10. Standard 与 Persona

### Standard

模型直接看到完整购物要求，主要测试：

- 查询构造；
- 搜索结果筛选；
- 商品比较；
- 规格选择；
- 购买决策。

配置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
```

### Persona

模型看到简化购物要求，并在 system message 中获得用户画像，需要从画像推断隐藏偏好。

配置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
  if_persona: true
```

两种模式使用相同的动作循环和原始 Reward，差异只在模型可见的任务信息。

## 11. 动作格式

模型每一步必须且只能发起一次标准 tool call。单 Agent 模式提供：

```json
{"name": "search", "arguments": {"keywords": "商品关键词"}}
```

或者：

```json
{"name": "click", "arguments": {"value": "当前可点击值"}}
```

多方交互模式还提供：

```json
{"name": "ask_shopper", "arguments": {"question": "需要确认的问题"}}
```

`tool_adapter.py` 会将工具调用转换为环境原有的 `search[...]` / `click[...]`
动作。下一步所需的环境 observation 会以带有对应 `tool_call_id` 的
`role: tool` 消息加入模型上下文；终止状态和 Reward 等结构化信息只在程序
内部处理。购买前仍然必须至少选择一个商品规格。

## 12. 输出文件结构

每个任务生成一个 JSON：

```json
{
  "task_id": 0,
  "reward": 1.0,
  "reward_detail": {
    "r_type": 1.0,
    "r_att": 1.0,
    "r_option": 1.0,
    "r_price": true
  },
  "goal": {},
  "purchase": {},
  "conversation": []
}
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `reward` | ShopSimulator 原始综合 Reward |
| `reward_detail` | 类别、属性、规格、价格等子分数 |
| `goal` | 数据集目标商品及约束 |
| `purchase` | Agent 实际购买的商品与规格 |
| `conversation` | 完整模型交互轨迹 |

## 13. 搜索配置

当前默认搜索实现是：

```text
SQLite FTS5 + 中文 bigram + 多字段 BM25
```

字段权重定义在：

```text
ShopSimulator/shop_env/web_agent_site/engine/search.py
```

默认权重：

```python
{
    "title": 3.0,
    "brand": 2.0,
    "category": 2.0,
    "model": 2.5,
    "attributes": 1.5,
    "options": 1.2,
    "bullets": 0.8,
}
```

搜索索引不包含任务 goal、reward 或答案字段，防止检索阶段泄漏目标答案。

将旧搜索索引移到备份文件后，下一次启动会自动重建：

```bash
mv ShopSimulator/shop_env/search_engine/products.sqlite3 \
   ShopSimulator/shop_env/search_engine/products.sqlite3.bak
```

只有在确认需要重建时才执行此命令。

## 14. 常见问题

### 无法连接 5000 端口

错误示例：

```text
Connection refused: 127.0.0.1:5000
```

处理：先在终端 A 启动 `pack_api.py`，并保持该进程运行。

### 找不到商品数据

错误示例：

```text
No such file or directory: items_eval_train.json
```

处理：按照“商品数据”章节解压文件并创建软链接。

### 找不到中文 spaCy 模型

错误示例：

```text
Can't find model 'zh_core_web_sm'
```

处理：安装 `zh_core_web_sm-3.7.0` wheel，并用下面的命令验证：

```bash
python -c "import spacy; spacy.load('zh_core_web_sm'); print('OK')"
```

### API Key 未读取

确认 `.env` 位于仓库根目录，而不是 `single_eval` 目录：

```text
LongHorizonAgentRL/.env
```

同时确认配置使用：

```yaml
model_key_env: DEEPSEEK_API_KEY
```

### 显示所有任务都已完成

`single_eval` 会根据输出目录中的 `<task_id>.json` 自动断点续跑。更换 `output_path`，或者备份旧结果后再运行。

### DeepSeek 返回空 content

DeepSeek 当前模型默认可能启用思考模式。评测配置中设置：

```yaml
thinking: disabled
```

确保有限的输出 token 用于生成可执行的 `Thought` 和 `Action`。
