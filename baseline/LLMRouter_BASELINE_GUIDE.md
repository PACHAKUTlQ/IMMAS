# IceMAS baseline：LLMRouter 使用与本地 vLLM 替换指南（简明版）

本文面向：你已经把 `baseline/LLMRouter` 作为对照组（baseline）放进 IceMAS 工程，希望：
1) 用 LLMRouter 的路由器/训练/推理/评测流程跑出一组可复现 baseline；
2) 把默认的“远程 API 调用”切换为“本地 vLLM 部署模型”。

---

## 1. baseline 用 LLMRouter 的推荐方式

把 LLMRouter 当作 baseline 的关键是“固定可复现的协议”：

- **固定候选模型池（LLM candidates）**：由 `default_llm.json` 定义模型集合、价格、服务商、以及 `api_endpoint`。
- **固定数据与拆分**：使用 LLMRouter 的 data pipeline（或你自己的数据），确保 train/test 划分与随机种子一致。
- **固定路由器与配置**：选择 1~2 个代表性路由器（例如 `smallest_llm` / `largest_llm` 作为下界/上界，外加一个可训练路由器如 `svmrouter`/`mlprouter`）。
- **固定推理参数**：如 `max_tokens/temperature/top_p` 等，避免“同模型不同采样策略”引入干扰。

你可以把 baseline 定义为：
- 下界：`smallest_llm`（永远选最小模型）
- 上界：`largest_llm`（永远选最大模型）
- 学习型：`svmrouter` 或 `mlprouter`

这些路由器配置示例在：`baseline/LLMRouter/configs/model_config_{train,test}/`。

---

## 2. LLMRouter 的目录与调用链（你需要理解的最少部分）

- 模型候选与 endpoint：`baseline/LLMRouter/data/**/llm_candidates/default_llm.json`
- 数据生成与评测（会进行模型调用）：`baseline/LLMRouter/llmrouter/data/api_calling_evaluation.py`
- 统一 API 调用封装（推荐关注）：`baseline/LLMRouter/llmrouter/utils/api_calling.py`
  - 通过 **LiteLLM** 调用 OpenAI 兼容接口：`completion(model="openai/<api_name>", api_base=<api_endpoint>)`
  - 支持 `API_KEYS` 环境变量做手动轮询与多服务商 key 管理
- 少数地方直接使用 `openai` SDK（需要一起切到本地 base_url）：
  - `baseline/LLMRouter/llmrouter/utils/evaluation.py`（`model_prompting()`）
  - `baseline/LLMRouter/llmrouter/models/automix/data_pipeline.py`
  - `baseline/LLMRouter/llmrouter/models/router_r1/route_service.py`

结论：
- **只要你让“OpenAI 兼容 base_url”指向 vLLM**，大部分路径无需改代码即可从远程 API 切到本地。

---

## 3. 作为 baseline 的最小复现实验流程

LLMRouter 官方 pipeline 是三步（建议先跑通 example data 再替换成你自己的数据）：

1) 生成 query 数据
- 入口：`baseline/LLMRouter/llmrouter/data/data_generation.py`

2) 生成 LLM 候选 embedding
- 入口：`baseline/LLMRouter/llmrouter/data/generate_llm_embeddings.py`

3) 调用候选模型 + 评测 + 生成 routing data
- 入口：`baseline/LLMRouter/llmrouter/data/api_calling_evaluation.py`
- 这一步会真正打模型（默认走 API），生成训练路由器需要的 routing 标注数据。

示例配置文件在：`baseline/LLMRouter/llmrouter/data/sample_config.yaml`。

---

## 4. 从远程 API 切换为本地 vLLM（推荐：OpenAI 兼容 Server 方案）

### 4.1 启动 vLLM OpenAI 兼容服务

常见方式（示例命令，按你机器/模型路径调整）：

```bash
# 方式 A：vLLM 新版常见命令（若你的环境支持）
vllm serve /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name local-llm

# 方式 B：OpenAI API server 入口（某些版本仍使用该形式）
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name local-llm
```

启动后，OpenAI 兼容 base_url 通常为：
- `http://127.0.0.1:8000/v1`

你可以用 `curl` 验证（可选）：

```bash
curl http://127.0.0.1:8000/v1/models
```

### 4.2 修改 LLM candidates：把 api_endpoint 指向本地 vLLM

LLMRouter 的关键配置文件是 `default_llm.json`（示例文件位于 `baseline/LLMRouter/data/example_data/llm_candidates/default_llm.json`）。

你需要做三件事：

1) 把每个候选模型的 `api_endpoint` 改为本地：
- `"api_endpoint": "http://127.0.0.1:8000/v1"`

2) 把每个候选模型的 `model`（即 `api_name`）改为 vLLM 暴露的模型名
- 推荐用 `--served-model-name local-llm` 固定一个短名字，然后在 JSON 中写 `"model": "local-llm"`

3) 建议把 `service` 改为 `Local`（便于你用 `API_KEYS` dict 格式配置空 key）

一个最小示例（你可以只保留 1~2 个候选模型作为 baseline pool）：

```json
{
  "local-7b": {
    "size": "7B",
    "feature": "Local vLLM served model",
    "input_price": 0.0,
    "output_price": 0.0,
    "model": "local-llm",
    "service": "Local",
    "api_endpoint": "http://127.0.0.1:8000/v1"
  }
}
```

### 4.3 配置 API_KEYS：本地 vLLM 允许空 key

LLMRouter 的 `llmrouter/utils/api_calling.py` 支持对 localhost endpoint 使用空 key，但 **推荐用 dict 格式**（因为 list 格式会过滤掉空字符串）。

```bash
export API_KEYS='{"Local": ""}'
```

对应地，你的 `default_llm.json` 里 `"service": "Local"` 需要匹配上。

### 4.4 同步影响：少数代码路径直接用 openai SDK

如果你会走到以下模块（例如某些评测/automix/router_r1），还需要设置：

```bash
export OPENAI_API_BASE='http://127.0.0.1:8000/v1'
# vLLM 通常不校验 key，但 openai SDK 需要一个非空字符串
export OPENAI_API_KEY='local'
```

这样 `baseline/LLMRouter/llmrouter/utils/evaluation.py` 的 `model_prompting()` 会把请求打到 vLLM。

---

## 5. 另一种替换方案（可选）：直接调用 vLLM Python API

不推荐作为“baseline 对照”第一选择，原因：
- 需要写适配层，把 LLMRouter 期望的 OpenAI-chat-completions 形式转成 vLLM 的 generate；
- 会绕过 LLMRouter 已经统一好的 LiteLLM/OpenAI-compatible 调用链。

只有当你明确不想跑 HTTP server、或者需要极致性能时，再考虑。

---

## 6. 常见坑与排错清单

- **base_url 是否包含 `/v1`**：LLMRouter 的 `api_endpoint` 示例以 `/v1` 结尾；vLLM 的 OpenAI 兼容也通常使用 `...:8000/v1`。
- **JSON 里的 `model` 名称要和 vLLM 的 served model name 一致**：不一致会导致 404/"model not found"。
- **空 key 的写法**：建议 `API_KEYS` 用 dict 格式 `{"Local":""}`，并确保 `service` 字段一致。
- **并发与超时**：`api_calling_evaluation.py` 里 `--workers` 很大时，本地单卡模型可能会排队/超时；先用小并发（例如 4~16）跑通。
- **token 上限**：vLLM 侧的 max_model_len / kv cache 设置会影响长输入任务。

---

## 7. baseline 汇报建议（写论文/报告时怎么说）

建议你在实验设置里明确：
- 候选模型池（数量、名称、参数量）、本地部署方式（vLLM，GPU 型号，batching/并发策略）
- 路由器种类（smallest/largest/学习型）与训练数据来源
- 推理超参（temperature/top_p/max_tokens）
- 评测指标与数据集

---

如果你希望我顺手把你现在工程里实际使用的候选模型 JSON（或你自己的模型池）改成 vLLM 本地配置，并给出一套可直接跑的命令组合（含并发建议），告诉我你打算部署的模型路径/名称与端口即可。
