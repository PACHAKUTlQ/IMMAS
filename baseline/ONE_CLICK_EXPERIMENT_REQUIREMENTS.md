# One-click 在线路由实验：需求规格（v0）

本文是“实现一键实验脚本”的需求文档，用于对齐目标、接口、产物与待确认点。

目标场景：通过本地 vLLM 同时部署多个大模型，持续接收请求流（requests），由 LLMRouter（baseline）执行“在线路由”（router online routing），评估 **many requests → many LLMs** 的在线表现，并沉淀完整日志与可复现结果。

---

## 0. 背景与目标

### 0.1 你希望解决的问题
- 当前 LLMRouter 多数 router 的调用假设是 **one request → many LLMs**（对一个 query 在候选池里多模型调用/比较/打分）。
- 你的研究目标是 **many requests → many LLMs**（在线流式请求，批处理路由，观察吞吐、延迟、失败率、成本等在线指标），并与后续你自己的 many-to-many router 对比。

### 0.2 本次交付物（最终要落地）
- 一个 `*.sh` 一键运行脚本：执行端到端实验（启动 vLLM / 运行路由 / 生成结果与图）。
- 实验可同时使用：
  - 多轮对话数据集（请求包含历史多轮 + 新问题）
  - 兼容原有单轮数据集（不移除）
- 全量日志与产物保存在 baseline 下（便于 debug、复现、归档）。
- 具备可扩展性：后续追加新路由器 / 新数据集 / 新指标 / 新图表尽量不推翻现有结构。

---

## 1. 术语定义

- **Request**：一次线上请求，包含当前输入（可能含对话历史）以及必要的元数据（request_id、timestamp、dataset、turn_id 等）。
- **Batch**：按“时间窗”或“数量阈值”聚合的一组 Requests，作为一次路由调度单元。
- **Router online**：路由器已经训练完成（如需训练），实验只评估其在线路由阶段。
- **One-to-many baseline**：LLMRouter 现有实现常见形态：单 request 内对多个 LLM 候选做调用/打分。
- **Many-to-many baseline（拼接法）**：对 batch 中每个 request 分别走一次 one-to-many，然后把结果拼起来，形成 many-to-many 行为。

---

## 2. 功能需求（Must/Should）

### R1（Must）多轮 + 单轮数据集兼容
- 支持多轮对话数据集（例如 CoQA/QuAC/ShARC/INSCIT）：
  - 一个 request 的输入包含：前面若干轮（history）+ 当前用户问题。
  - 需要能控制“包含多少轮历史”（例如 `history_turns=0/1/3/all`）。
- 保留并继续支持原有单轮数据（LLMRouter 当前 11 个基准任务或你已有 JSONL）。
- 数据入口形式要求：
  - 优先统一为 JSONL 流（每行一个 request），便于流式读取与在线 batch。

### R2（Must）只评估训练后在线表现
- 对需要训练的 router：
  - 训练阶段可以单独运行，但在线实验必须使用训练完成的 checkpoint。
  - one-click 脚本可以“可选地先训练”，但默认应支持“直接加载 checkpoint 做在线实验”。

### R3（Must）Batching：收集后再路由
- 请求进入系统后，先在 buffer 中累积。
- 满足任一条件触发一次 batch：
  - 时间窗到期（例如每 200ms/1s flush）
  - 数量阈值到达（例如 16/64 requests）
- 路由器对 batch 进行分配。
- baseline many-to-many 实现方式：
  - 对 batch 内每条 request 调用一次 one-to-many router（顺序或并发），把结果聚合成 batch 结果。

### R4（Must）尽可能多的日志与产物（debug 优先）
- 实验必须输出：
  - 单个统一日志文件（包含环境、配置、每个 batch 的关键事件、错误栈）。
  - 结构化结果（CSV/JSONL）与可视化图表（PNG/SVG）。
- 日志与结果统一落在 baseline 下某个 run 目录，并可按时间戳/实验名区分。

### R5（Should）可发展性
- 采用“配置驱动 + 插件化/可扩展接口”，避免硬编码。
- 要能在后续方便添加：
  - 新数据集适配器
  - 新 router（尤其是你自己的 many-to-many router）
  - 新指标与新图表

---

## 3. 一键脚本的行为规范（建议）

### 3.1 统一入口
- 建议提供一个脚本作为唯一入口，例如：
  - `baseline/run_one_click_experiment.sh`

该脚本应按顺序完成：
1) 检查环境与依赖（Python env、CUDA、端口占用、模型路径）
2) 启动/复用 vLLM 服务（多个模型、多个端口）
3) 运行在线路由实验主程序
4) 汇总并生成图表
5) 输出最终结果索引（run 目录位置、关键指标）

### 3.2 配置文件
- one-click 脚本必须支持指定一个配置文件（YAML/JSON）
- 配置至少包含：
  - vLLM served models 列表（name、model_path、port、tensor_parallel 等）
  - 数据源（单轮/多轮、路径、抽样量、history_turns、stream 模式）
  - batch 策略（time_window_ms、max_batch_size）
  - router 选择与 checkpoint 路径
  - 并发策略（每模型并发、总并发、超时）
  - 输出目录

---

## 4. 在线实验流程（建议的系统分层）

### 4.1 组件划分
1) **Dataset Adapter**
   - 输入：HF dataset / 本地 JSON
   - 输出：统一的 request 流（JSONL）

2) **Request Streamer**
   - 以固定速率/泊松到达/回放 trace 的方式投喂 request（用于控制负载形态）。

3) **Batcher**
   - 维护 buffer
   - 根据 time/size 触发 flush

4) **Router Runner（baseline 模式）**
   - 对每条 request 调用 one-to-many router（LLMRouter 原有）
   - 聚合成 many-to-many（拼接法）

5) **Metrics & Reporter**
   - 在线指标：latency（p50/p95/p99）、throughput、错误率、token 使用、队列长度、分配比例等
   - 任务指标：EM/F1/等（取决于 dataset）

6) **Artifact Writer**
   - 落盘 JSONL/CSV
   - 生成图表

### 4.2 vLLM 部署约束
- 每个大模型一个 vLLM 服务实例（通常：一个端口对应一个模型）。
- 与 LLMRouter 对接方式优先采用 OpenAI 兼容接口（`http://127.0.0.1:<port>/v1`）。

---

## 5. 结果与日志规范（建议的目录结构）

建议每次实验生成一个 run 目录：

- baseline/logs/<run_id>/
  - run_config.yaml（最终生效配置，便于复现）
  - run.log（完整日志）
  - requests.jsonl（输入请求回放/采样）
  - routing_decisions.jsonl（每条 request 的路由结果与原因）
  - model_calls.jsonl（每次对 LLM 的调用：prompt/token/latency/错误）
  - metrics.csv（按时间片或按 batch 统计）
  - summary.json（最终汇总指标）
  - figures/（所有图表）

日志字段建议（至少）：
- run_id、timestamp、event_type
- request_id、dataset、conversation_id、turn_id
- batch_id、batch_size、queue_wait_ms
- chosen_model、candidate_models（如可得）
- llm_latency_ms、router_latency_ms、end_to_end_ms
- prompt_tokens/completion_tokens/total_tokens（若可得）
- exception_type/stacktrace（失败时）

---

## 6. 评估指标（需要明确口径）

### 6.1 在线系统指标（强烈建议必须有）
- E2E latency：request 从进入系统到拿到最终答复的耗时
- Router latency：路由决策耗时（含批处理）
- LLM latency：模型生成耗时
- Throughput：req/s、tok/s
- Error rate：超时、HTTP 错误、OOM、空回复
- Load balance：不同 LLM 被分配比例、模型繁忙度

### 6.2 任务正确性指标（可选，但你前面关心 metric）
- 单轮 QA：cem/em/f1/em_mc 等（LLMRouter 已有）
- 多轮对话：建议先复用 f1/em（对每个 turn 的 answer），但 ShARC/INSCIT 可能需要自定义 metric。

---

## 7. 与当前 LLMRouter 代码的对接点（现状）

已知（baseline 内）：
- LLMRouter 的评估函数集中在：
  - [baseline/LLMRouter/llmrouter/utils/evaluation.py](baseline/LLMRouter/llmrouter/utils/evaluation.py)
  - [baseline/LLMRouter/llmrouter/evaluation/batch_evaluator.py](baseline/LLMRouter/llmrouter/evaluation/batch_evaluator.py)
- API 调用有两套路径：
  - LiteLLM 统一调用（更适合对接 vLLM OpenAI 兼容服务）
  - 少量 openai SDK 直接调用（需要 base_url 指向本地）

---

## 8. 存疑点（需要你确认/选择）

为保证实现不偏离你的真实实验设定，以下问题需要明确：

1) **在线请求“到达过程”是什么？**
   - A. 固定 QPS（例如 50 req/s）
   - B. 泊松过程（更贴近线上）
   - C. trace 回放（从已有时间戳日志回放）

这一阶段选择B

2) **batch 触发策略优先级？**
   - A. 先到 size 立刻 flush，否则等到 time
   - B. 先到 time 立刻 flush，即使 batch 很小

这一阶段选择到达size flush，否则等到time，在time时及时batch很小也flush

3) **多轮请求的 history 包含哪些内容？**
   - A. 仅用户问题（user-only）
   - B. user+assistant 全历史（推荐）
   - C. 截断到最后 N 轮（N 可配）
   - D. 是否包含 router 选择的模型名/工具调用等元信息？（通常不包含）

 B

4) **多轮任务的 ground_truth 与 metric 口径？**
   - CoQA/QuAC 可用 EM/F1
   - ShARC/INSCIT 是否也要算 correctness？还是只算在线系统指标？
 
 CoQA/QuAC用 EM/F1，另外两个只算在线系统指标（未来保留计算的可能性）


5) **LLMRouter baseline 的 many-to-many（拼接法）并发策略**
   - A. batch 内每条 request 串行处理（最简单但吞吐低）
   - B. batch 内并发调用（更符合线上，但更复杂）
   - C. 并发上限如何设定（全局/每模型/每端口）

    B， 并发上限设定为每模型

6) **vLLM 部署方式与资源约束**
   - 一个 GPU 跑几个模型？是否允许多 GPU？
   - 每个模型的参数（tp、max_model_len、gpu_memory_utilization）是否需要写进配置？

   自动选择显存足够的GPU部署，每个模型的参数写进配置

7) **“最终输出”定义**
   - 路由输出只需要 chosen model + response？
   - 是否还要保留候选模型的对比结果（one-to-many 会产生）？

    需要保留候选模型的对比结果
---

## 9. 非目标（本期不做/可后续扩展）

- 不追求把 LLMRouter 内部所有 router 全部改成原生 many-to-many；baseline 仅采用“拼接法”。
- 不强制实现 INSCIT 的 grounded evidence 评测（除非你要求）。
- 不默认做大规模分布式压测（多机）。

---

## 10. 下一步（建议的开发顺序）

1) 明确第 8 节存疑点的选择
2) 固化 run 目录结构与日志字段
3) 先跑通：单轮数据 + 单模型 vLLM + smallest/largest router
4) 再扩展：多模型 vLLM + batching + 多轮数据
5) 最后：引入你的 many-to-many router 做对比
