# IceMAS Baseline 交接文档（仅 baseline 范围）

本文面向：接手 baseline（LLMRouter + one_click 在线实验）的人，目标是用最少时间跑通实验、看懂产物、并能接入新的 router。

---

## 1) 目前能做到 / 不能做到

### 能做到（已具备）

- **一键在线实验（many requests → many LLMs）**
  - 入口脚本：`baseline/run_one_click_experiment.sh`
  - 支持请求流（Poisson arrival）、batching（size_then_time）、并发控制（每模型并发）。
- **本地 vLLM（OpenAI-compatible）替换远程 API**
  - one_click 可按配置启动多个 vLLM server（多端口、多模型），或复用已启动 server。
- **多轮数据（CoQA/QuAC）自动准备 + JSONL 展平**
  - one_click 在运行时发现缺失的 `coqa_*_multiturn.jsonl` / `quac_*_multiturn.jsonl` 会自动下载并生成（可通过环境变量控制生成规模）。
- **Shadow compare（保留候选模型对比结果）**
  - 可在 router 选出 chosen 模型外，对其它候选做 shadow 调用，保存对比信息。
- **日志与可复现产物落盘**
  - 每次 run 写入 `baseline/logs/<run_id>/`，包含：配置、requests、路由决策、模型调用、metrics.csv、summary.json、figures。
- **任务 performance + token + social welfare 的记录/汇总/作图（chosen 维度）**
  - `task_performance`：基于模型输出 vs `ground_truth` 的评测（例如 CoQA/QuAC 常用 EM/F1）。
  - token：来自模型调用返回的 `prompt_tokens/completion_tokens/total_tokens`。
  - social welfare：按 chosen 调用计算并汇总（定义见 `summary.json` 的 `chosen_social_welfare.definition`）。

### 不能做到（当前限制/非目标）

- **真正“联合优化”的 many-to-many 路由（batch 内统一分配）**
  - 目前 one_click 的在线实验是对 batch 内每条 request 逐条调用 `router.route_single()`，属于“拼接式 many-to-many”。
  - 你可以实现 `route_batch()`，但当前 one_click 默认不会调用它来做联合分配（需要后续扩展 one_click 才能原生支持）。
- **ShARC/INSCIT 的 correctness 评测**
  - 当前 baseline 主要把 ShARC/INSCIT 当作“在线系统指标评估”任务（correctness 指标保留未来扩展空间）。
- **分布式/多机压测**
  - 当前是单机多端口/多 GPU 的实验框架，不包含多机压测与分布式调度。

---

## 2) 如何使用（最少步骤）

### 2.1 直接跑一个 random demo（推荐）

- 直接运行：

```bash
cd /home/lhz/lyz/IceMAS
bash baseline/run_demo.sh
```

`baseline/run_demo.sh` 会调用 one_click 的配置（见脚本内引用的 yaml），并生成：
- `baseline/logs/<run_id>/summary.json`
- `baseline/logs/<run_id>/figures/*.png`

### 2.2 自己指定配置跑 one_click

- 例：smoke（少量请求）

```bash
cd /home/lhz/lyz/IceMAS
RUN_ID=smoke_$(date +%Y%m%d_%H%M%S) \
  bash baseline/run_one_click_experiment.sh baseline/one_click/configs/smoke_coqa_random.yaml
```

- 例：多请求 random（复用已启动 vLLM）

```bash
cd /home/lhz/lyz/IceMAS
RUN_ID=verify_random_$(date +%Y%m%d_%H%M%S) \
  bash baseline/run_one_click_experiment.sh baseline/one_click/configs/verify_random_token_welfare.yaml
```

### 2.3 输出产物怎么查看

每次 run 的目录：
- `baseline/logs/<run_id>/`

建议优先看：
- `summary.json`：run 级汇总指标（含 chosen 的 perf/token/welfare 与 latency p50/p95 等）
- `figures/`：自动生成的图
- `model_calls.jsonl`：逐次模型调用（含 tokens、task_performance、social_welfare）
- `routing_decisions.jsonl`：路由决策（chosen 模型与 router 输出信息）
- `metrics.csv`：按 batch 写的运行时指标（吞吐/错误/滚动统计等）

### 2.4 常用环境变量（脚本已做默认兜底）

`baseline/run_one_click_experiment.sh` 默认会设置/兜底：
- `HF_ENDPOINT` / `HF_HUB_ENDPOINT`：指向 HF mirror（网络受限时必需）
- `OPENAI_API_KEY` / `API_KEYS`：本地 vLLM/LiteLLM 调用也需要非空 key

如果你本机网络正常，也可以自行覆盖这些变量。

---

## 3) 配置参数怎么改（one_click 配置速查）

one_click 的实验配置在：`baseline/one_click/configs/*.yaml`。

### 3.1 运行模式

- `skip_vllm: true|false`
  - `false`：one_click 会按 `vllm:` 配置自动启动本地 vLLM（端口/模型/显卡自动选择）。
  - `true`：不启动 vLLM，直接假设 `llm_candidates_path` 里的 endpoints 已经跑着。
- `dry_run: true|false`
  - `true`：不请求任何 LLM，只跑 arrival/batching/日志落盘（用于验证管线和产物结构）。

### 3.2 vLLM 启动参数（当 `skip_vllm: false`）

位置：`vllm.models[*]`。

- `model_path`：HuggingFace repo id 或本地路径（例如 `Qwen/Qwen2.5-7B-Instruct`）。
- `served_model_name`：vLLM 对外暴露的模型名（要和 `llm_candidates_path` 里 `api_name/model` 对上）。
- `port`：每个模型一个端口（OpenAI-compatible，通常 base URL 为 `http://127.0.0.1:<port>/v1`）。
- `gpu: auto|<int>`：`auto` 会挑显存足够的 GPU；也可手动 pin 到指定 GPU。
- `tensor_parallel_size` / `max_model_len` / `gpu_memory_utilization` / `max_num_seqs`：vLLM 性能与稳定性相关参数。
- `attention_backend` / `enforce_eager`：为稳定性/兼容性预留的开关（某些环境下必需）。
- `per_model_max_concurrency`：**one_click 每模型并发上限**（影响在线实验吞吐与排队；不是 vLLM 内部并发）。

### 3.3 候选模型池（LLM candidates）怎么改

- one_click 的候选池文件：`baseline/one_click/llm_candidates/vllm_candidates.json`
  - 每个候选包含：`api_endpoint`（指向 vLLM）、`model`/`api_name`（对应 served_model_name）。
- 你也可以为不同实验写不同候选池 JSON，然后在实验配置里改：
  - `llm_candidates_path: <your_json>`

#### 3.3.1 如何新增/修改一个 LLM（最常用）

要让 one_click “能选到/能调用到”一个新 LLM，需要两件事对上：

1) **有一个可用的 OpenAI-compatible endpoint**（本地 vLLM 或远程兼容服务）
2) **把它写进候选池 JSON**（让 router & one_click 知道它的名字和 endpoint）

最常见两种方式：

**方式 A：新增本地 vLLM 模型（推荐）**

- 在实验 config 里加一条 `vllm.models[*]`（当 `skip_vllm: false`）：
  - 关键字段：`model_path`、`served_model_name`、`port`
- 同时更新候选池 JSON（通常直接改 `baseline/one_click/llm_candidates/vllm_candidates.json`）：
  - `api_endpoint`: `http://127.0.0.1:<port>/v1`
  - `model`: 必须等于 `served_model_name`
  - key 名（例如 `"Mistral-7B"`）是候选“显示名/路由名”，router 输出的 `model_name` 需要匹配这个 key。

**方式 B：接入一个远程 OpenAI-compatible 服务**

- 不需要写 `vllm:`，直接在候选池 JSON 里新增一项：
  - `api_endpoint`: 远程 base url（以 `/v1` 结尾）
  - `model`: 远程服务里可用的 model id
  - 可选：`service`（用于从 `API_KEYS` 里选 key；本地一般不需要复杂配置）

#### 3.3.2 候选池 JSON 的字段约定（one_click 用到哪些）

以 `baseline/one_click/llm_candidates/vllm_candidates.json` 为准，one_click 读取并使用：
- `api_endpoint`：OpenAI-compatible base URL（建议带 `/v1`）
- `model`：传给 LiteLLM/OpenAI 的 model 名（需与服务端一致）
- 其余字段（如 `service`）用于 LLMRouter/LiteLLM 的 provider/key 选择（本地 vLLM 通常不用改）。

#### 3.3.3 常见踩坑（新增 LLM 时最容易错）

- `served_model_name`、候选池 JSON 的 `model`、以及 vLLM 实际暴露的 model 名不一致 → 直接 404/"model not found"。
- router 输出的 `model_name` 必须等于候选池 JSON 的 key（例如 `"Mistral-7B"`），否则 one_click 会提示 chosen 不在 candidates。
- endpoint 有鉴权（`/v1/models` 返回 401）→ 确保 `OPENAI_API_KEY`/`API_KEYS` 非空，且请求带 Authorization（脚本已默认兜底）。

### 3.4 请求流与数据集参数

位置：`request_stream:`。

- `arrival_process: poisson`
- `poisson_qps`：请求到达强度（越大压力越高）。
- `max_requests` / `max_duration_sec`：终止条件。
- `datasets[*]`：可混合多个数据集（按 `weight` 抽样）。
  - `type: jsonl` + `path`：一行一个 request。
  - `history_mode` / `history_turns`：多轮拼接方式。
  - `metric_mode`：对多轮数据生成时指定 metric（常见 `em_f1`）。

### 3.5 Batching 与执行参数

- `batching.max_batch_size`：触发 flush 的 size 阈值。
- `batching.time_window_ms`：触发 flush 的时间窗。
- `batching.flush_policy`：当前主要用 `size_then_time`。

执行参数（影响生成与超时）：
- `execution.timeout_sec`
- `execution.max_tokens`
- `execution.temperature`

### 3.6 Shadow compare（保留候选对比）

位置：`shadow_compare:`。

- `enabled: true|false`
  - `true`：chosen 之外还会对其它候选做 shadow 调用（同题同参，便于公平对比）。
- `max_candidates_per_request: all|<int>`：控制 shadow 的候选数量。

### 3.7 Social welfare 配置（基于 performance 与 token）

位置：`social_welfare:`。

- `token_cost_field: total_tokens|prompt_tokens|completion_tokens`
- `token_cost_weight: <float>`

当前 welfare 定义（chosen calls only）：
- `welfare_i = task_performance_i - token_cost_weight * token_cost_i`

如果你希望 welfare 的量纲更合理（因为 performance ∈ [0,1]，tokens 通常几百）：建议把 `token_cost_weight` 设为 `1e-3`（即“每 1k tokens 扣 1 分”一类的尺度）。

### 3.8 输出控制（落盘哪些文件/图）

位置：`output:`。

- `run_dir: null|<path>`：默认由脚本创建 `baseline/logs/<run_id>/`；你也可以显式指定一个目录。
- `write_requests_jsonl` / `write_model_calls_jsonl` / `write_routing_decisions_jsonl`
- `write_metrics_csv` / `write_summary_json` / `write_figures`

调试时建议全开；极限压测时可以关闭部分输出减少 IO。

### 3.9 多轮数据自动准备的参数（CoQA/QuAC）

`baseline/run_one_click_experiment.sh` 会在运行前检查 config 中引用的多轮 JSONL 是否存在；缺失时会调用：
- `baseline/one_click/data/build_multiturn_jsonl.py`

常用环境变量：
- `MAX_CONVERSATIONS=0`：生成全量对话（设为 50/200 可做 smoke）
- `MAX_TURNS_PER_CONV=0`：保留全量 turns（设为 3/5 可加速）
- `HISTORY_TURNS=all`：history 拼接策略（通常保持 all）

### 3.10 one_click 实验 YAML 写法（`baseline/one_click/configs/*.yaml`）

one_click 的“实验配置 YAML”是端到端运行的配置：启动 vLLM（可选）+ 请求流 + batching + 执行参数 + router + 输出。

最推荐从这些现成文件复制再改：
- `baseline/one_click/configs/smoke_coqa_random.yaml`（最小可跑）
- `baseline/one_click/configs/random_demo.yaml`（多请求 demo/对比用）
- `baseline/one_click/configs/compare_coqa_val_randomrouter.yaml`（compare 模式示例）

一个“最小模板”（字段含义见下面注释；可直接复制作为新实验）：

```yaml
skip_vllm: true            # true: 复用已有 vLLM；false: 按 vllm: 启动
dry_run: false             # true: 不打模型，仅验证管线

vllm:                      # 仅 skip_vllm:false 时生效
  host: "127.0.0.1"
  launch_method: "auto"   # auto/vllm_serve/python_api_server
  models:
    - name: "Mistral-7B"  # 候选池的 key 名（路由输出也要匹配）
      model_path: "mistralai/Mistral-7B-Instruct-v0.3"
      served_model_name: "mistral-7b"   # vLLM 实际 model id
      port: 8012
      gpu: auto
      tensor_parallel_size: 1
      max_model_len: 4096
      gpu_memory_utilization: 0.90
      enforce_eager: true
      attention_backend: "TRITON_ATTN"
      per_model_max_concurrency: 4        # one_click 每模型并发上限

router:
  name: "randomrouter"    # LLMRouter 的 router 名称（内置或插件）
  config_path: "baseline/one_click/router_configs/randomrouter_online.yaml"
  load_model_path: null    # 需要加载 checkpoint 的 router 可填路径

llm_candidates_path: "baseline/one_click/llm_candidates/vllm_candidates.json"

request_stream:
  arrival_process: "poisson"
  poisson_qps: 1.0
  max_requests: 20
  max_duration_sec: 600
  datasets:
    - name: "coqa_multiturn"
      type: "jsonl"
      path: "baseline/one_click/data/coqa_validation_multiturn.jsonl"
      weight: 1.0
      history_mode: "user_assistant"
      history_turns: "all"
      metric_mode: "em_f1"

batching:
  max_batch_size: 4
  time_window_ms: 500
  flush_policy: "size_then_time"

execution:
  timeout_sec: 60
  max_tokens: 256
  temperature: 0.2

shadow_compare:
  enabled: false           # true: 对所有候选做 shadow 调用（同题对比）
  max_candidates_per_request: "all"

social_welfare:
  token_cost_weight: 1.0
  token_cost_field: "total_tokens"

output:
  run_dir: null
  write_requests_jsonl: true
  write_model_calls_jsonl: true
  write_routing_decisions_jsonl: true
  write_metrics_csv: true
  write_summary_json: true
  write_figures: true
```

字段之间最重要的“对齐关系”是：
- `router.route_single()` 输出的 `model_name` 必须是 `llm_candidates_path` JSON 的 key
- `llm_candidates_path` JSON 的 `model` 必须和 vLLM 的 `served_model_name` 一致

---

## 4) 指标口径与“怎么改指标/加指标”

这里分两类指标：
- A) **任务正确性指标**（task performance）：需要 `ground_truth`，由评测函数计算。
- B) **系统在线指标**（latency/throughput/errors/tokens 等）：由 one_click 运行时统计。

### 4.1 task performance 是怎么计算的？

- 入口：`baseline/LLMRouter/llmrouter/utils/evaluation.py` 的 `calculate_task_performance(...)`
- one_click 调用位置：`baseline/one_click/online_experiment.py`（每次模型调用返回后，如果有 `ground_truth` + `metric` 则计算）。

注意：`metrics.csv` 记录的是按 batch 的运行时统计；**task_performance 不是从 metrics.csv “算出来的”**，而是从“模型输出 vs ground_truth”评测得来。

### 4.2 想新增/修改 task metric（例如新增一个新的 evaluation metric）

推荐方式：在 LLMRouter 的评测注册表里加一个 metric。

1) 在 `baseline/LLMRouter/llmrouter/evaluation/batch_evaluator.py` 里注册（`EVALUATION_METRICS` + 装饰器）。
2) 在数据 JSONL 里把 `metric` 字段设为你的 metric 名称（one_click 的请求会透传）。
3) 重新跑 one_click：`summary.json` 会自动聚合新的 task_performance 值（前提是你的 metric 返回 float）。

### 4.3 想新增/修改 one_click 的“实验汇总指标”（summary.json）

位置：`baseline/one_click/online_experiment.py`。

最常见的扩展方式是：
- 在 `ModelCall` 里新增字段（这样每次调用都能落到 `model_calls.jsonl`）。
- 在 run 内维护一个累积器（例如 `chosen_xxx: List[float]`）。
- 结束时把统计量写进 `summary.json`（通常至少写 `count/mean/p50/p95`）。

### 4.4 想新增/修改 one_click 的“作图”（figures/）

位置：同样在 `baseline/one_click/online_experiment.py`。

常见模式：
- 从 run 累积器拿到一组数（perf/tokens/welfare/latency…）
- 画 histogram/scatter
- 保存到 `baseline/logs/<run_id>/figures/`

### 4.5 想把新指标加入“多 run 对比表”

- 汇总脚本：`baseline/one_click/tools/summarize_compare_runs.py`
- 它是读取每个 run 的 `summary.json` 再打印 markdown 表。

你只需要在这个脚本里把你新增的 `summary.json` 字段取出来，加进 headers 即可。

### 4.6 想新增/修改 metrics.csv 的列（按 batch 的运行时指标）

位置：`baseline/one_click/online_experiment.py`。

思路：
- 先在 flush 里拿到你想统计的 batch 级信号（例如本 batch chosen 的 tokens、错误数、队列等待时间等）。
- 把列名加入 CSV header。
- 每次 flush 写一行数值。

注意：
- `metrics.csv` 适合放“随时间/随 batch 变化”的在线指标。
- `summary.json` 更适合放“run 结束后的整体统计”。

---

## 5) 如何添加新的 router（整合进当前项目）

这里分两层：
- A) **把 router 加进 LLMRouter 的插件系统**（让 `load_router` 能加载）
- B) **把 router 接到 one_click 在线实验**（用 one_click 的 config 指向它）

### 5.1 A：以“插件”的方式添加 router（推荐）

目录结构（示例）：

```
baseline/LLMRouter/custom_routers/my_joint_router/
  __init__.py
  router.py
  trainer.py   # 可选
  config.yaml  # router 自己的配置（LLM candidates 等）
```

要求：
- `router.py` 内必须有一个类名以 `Router` 结尾，并继承 `llmrouter.models.meta_router.MetaRouter`。
- 必须实现：
  - `route_single(self, query_input: dict) -> dict`
  - `route_batch(self, batch: list[dict]) -> list[dict]`
- 输出 dict 至少包含 `model_name`（one_click 会用它当 chosen 模型）。

可以直接参考：
- `baseline/LLMRouter/custom_routers/randomrouter/router.py`
- `baseline/LLMRouter/llmrouter/plugin_system.py`（插件发现与校验）

### 5.2 B：让 one_click 使用你的 router

one_click 的实验配置里会写：

```yaml
router:
  name: "<router_name>"
  config_path: "baseline/one_click/router_configs/<something>_online.yaml"
```

你需要新增一个 online 配置 YAML（给 LLMRouter 的 router 初始化用），放在：
- `baseline/one_click/router_configs/`

例如：
- `baseline/one_click/router_configs/my_joint_router_online.yaml`

它通常需要包含（最少）：
- `data_path.llm_data`：指向候选模型 JSON（one_click 默认用 `baseline/one_click/llm_candidates/vllm_candidates.json`）
- `hparam`：你的 router 超参

然后写一个 one_click 实验配置（放在 `baseline/one_click/configs/`），指定：
- 请求流（JSONL 数据）
- batching 与并发
- router.name 与 router.config_path

### 5.2.1 Router YAML config 写法（LLMRouter `yaml_path` 的格式）

这里说的“router 的 YAML config”，指的是 `router.config_path` 指向的文件（通常放在 `baseline/one_click/router_configs/*_online.yaml`），它会作为 `MetaRouter(yaml_path=...)` 的输入。

推荐直接参考现成文件：
- `baseline/one_click/router_configs/randomrouter_online.yaml`

常见结构（最少字段）：

```yaml
data_path:
  llm_data: '../one_click/llm_candidates/vllm_candidates.json'  # 候选模型池（注意相对路径基准）

metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0

hparam:
  # router 超参（自定义）
  seed: 42
```

要点：
- **路径基准**：`MetaRouter` 会以 `baseline/LLMRouter` 作为 project root 去解析相对路径，所以 online router config 里常用 `../one_click/...` 这种写法。
- `data_path.llm_data` 决定 router 能看到哪些候选模型（以及候选的元信息字段）。
- `hparam` 完全由你的 router 自己消费；你可以在里面放任何参数。
- `metric.weights` 是很多内置 router 的通用字段（有些 router 会用它把 performance/cost 组合成打分）。

### 5.2.2 修改/新增 router 的 config 怎么组织（建议约定）

- one_click 的 router online config 统一放在：`baseline/one_click/router_configs/`
  - 命名建议：`<router_name>_online.yaml`
- 你的自定义 router 如果需要自己的专用配置（例如额外数据文件），尽量写到 `hparam:` 或 `data_path:` 下，并保证相对路径按上述规则可解析。

### 5.3 关于“真正 many-to-many 联合路由”的注意事项

如果你的 router 需要“batch 内统一分配”（不是逐条独立决策），建议：
- 先实现 `route_batch()`，在 router 内完成联合分配并输出每条请求的 `model_name`。
- 但要注意：**当前 one_click 默认还是逐条调用 `route_single()`**。
  - 若要原生支持 batch 联合路由，需要后续对 one_click 做一个增强：在每次 flush 时调用 `router.route_batch()`。

---

## 附：baseline 入口文件索引

- 一键在线实验入口：`baseline/run_one_click_experiment.sh`
- demo 脚本：`baseline/run_demo.sh`
- one_click 说明：`baseline/one_click/README.md`
- LLMRouter baseline 指南：`baseline/LLMRouter_BASELINE_GUIDE.md`
- 多轮数据集指南：`baseline/MULTITURN_DATASETS_GUIDE.md`
- one_click 需求规格（背景/口径/日志结构）：`baseline/ONE_CLICK_EXPERIMENT_REQUIREMENTS.md`

---

## 附：常见问题排查（最短版）

- vLLM `/v1/models` 返回 401：说明 endpoint 开了鉴权；确保 `OPENAI_API_KEY`/`API_KEYS` 非空，并且请求带 Authorization（脚本已默认兜底）。
- 数据集下载失败（DNS/网络）：优先使用脚本默认的 `HF_ENDPOINT/HF_HUB_ENDPOINT`；或手动设置为可用 mirror。
- 一直超时/失败率高：先降低 `request_stream.poisson_qps`、降低 `execution.max_tokens`、降低 `per_model_max_concurrency`，再逐步加压。
- 显存不足：降低 vLLM 的 `max_model_len`、`gpu_memory_utilization`，或换更小模型。

