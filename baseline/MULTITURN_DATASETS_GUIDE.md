# 多轮对话数据集速览：CoQA / QuAC / ShARC / INSCIT（以及如何接入 LLMRouter baseline）

本文目标：
- 快速说明你给出的 4 个**多轮对话/信息寻求**数据集是什么、数据长什么样、怎么加载使用。
- 评估它们能否替换当前 `baseline/LLMRouter` 里默认的“11 个单轮基准数据集”。
- 给出**最省事**的接入方式（不大量改 LLMRouter 代码）。

---

## 1) 数据集是什么内容？（任务形式）

### CoQA（Conversational Question Answering）
- 链接： https://huggingface.co/datasets/stanfordnlp/coqa
- 核心任务：给定一段 passage（story/文章）和对话历史，逐轮提问，模型输出当前问题的答案（自由文本 + 证据 span 位置）。
- 规模（HF card）：约 7.2k train conversations；每个对话包含多轮 questions/answers。
- 许可证：来自多域文本的组合许可（Wikipedia/文学为 CC BY-SA 4.0 等，其他域有各自 license；在做再发布/商用时要留意）。

HF 上的字段（按一条对话为一个样本）：
- `story`: passage 文本
- `questions`: string list（一段对话的所有轮问题）
- `answers`: dict（每轮对应的 `input_text`、span start/end 等）
- `source`: domain（wikipedia/cnn/...）

### QuAC（Question Answering in Context）
- 链接： https://huggingface.co/datasets/allenai/quac
- 核心任务：信息寻求对话式 QA。给定 Wikipedia section 作为 context，对话进行多轮问答；答案多为 span（可出现 `CANNOTANSWER`）。
- 特点：每条数据以“dialogue”为单位，内部包含多轮 questions/answers，同时有 followup/yesno 等标签。
- 许可证：MIT（HF card）。

HF 上的字段（按一条对话为一个样本，示例字段）：
- `context`: Wikipedia section 文本
- `questions`: list
- `answers`: dict（span texts/offsets；dev/test 可能有多参考答案）
- `followups`, `yesnos`, `turn_ids` 等

注意：HF 页面提示该数据集 viewer 可能被禁用（因为 repo 有自定义加载脚本），但 `datasets.load_dataset("allenai/quac")` 通常仍可用。

### ShARC（Shaping Answers with Rules through Conversation）
- 链接： https://huggingface.co/datasets/UCLNLP/sharc
- 核心任务：**对话式机器阅读 + 规则解释**。
  - 给定规则片段（snippet）+ 场景描述（scenario）+ 对话历史（history 的 follow-up Q/A），模型需要输出：
    - 最终答案（常见是 yes/no/irrelevant 或需要继续问 follow-up question，具体以原论文/评测为准）
- 许可证：CC BY-SA 3.0（HF metadata）。
- 切分：train/validation（HF metadata 中 train 21890 / val 2270）。

HF metadata 暴露的主要字段：
- `snippet`: 规则文本
- `question`: 初始问题
- `scenario`: 场景描述
- `history`: list（`follow_up_question`/`follow_up_answer`）
- `answer`: string（目标输出）
- 以及 `negative_question`/`negative_scenario` 等标记

### INSCIT（Information-Seeking Conversations with Mixed-Initiative Interactions）
- 链接： https://github.com/ellenmellon/INSCIT
- 核心任务：信息寻求对话（多轮），并且带“证据 passages”。每个 turn 可能有 1~2 个标注（labels），包含：
  - `responseType`（回答类型）
  - `response`（agent 的回答文本）
  - `evidence`（引用证据 passage 列表）
- 数据格式：仓库 `./data` 内提供 train/dev/test（JSON）。
- 与 CoQA/QuAC/ShARC 不同：INSCIT 更像一个**带检索证据的对话生成/ grounded response**任务，复杂度更高。

---

## 2) 怎么用？（最小加载示例）

下面只给“怎么把数据读出来”的最小用法；具体训练/评测策略取决于你要做的任务设定。

### 2.1 Hugging Face datasets 方式（CoQA / QuAC / ShARC）

```python
from datasets import load_dataset

coqa = load_dataset("stanfordnlp/coqa")
# coqa["train"][0] 代表一条对话，里面 questions/answers 是列表

quac = load_dataset("allenai/quac")
# quac["train"][0] 同样是一条对话

sharc = load_dataset("UCLNLP/sharc")
# sharc["train"][0] 是一个对话状态样本（规则/场景/历史/答案）
```

### 2.2 GitHub 仓库数据（INSCIT）

INSCIT repo 里 `./data` 就是 json 文件（官方 README 有示例结构）。你可以直接 `json.load` 读入，然后按 turn 展平。

---

## 3) 能不能替换 LLMRouter baseline 里的数据集？

结论分三档：

- **可以替换，但不建议“直接把它们塞进现有 data_generation.py”**。
  - LLMRouter 的 [baseline/LLMRouter/llmrouter/data/data_generation.py](baseline/LLMRouter/llmrouter/data/data_generation.py) 是为“11 个单轮 benchmark”写死的抽样逻辑。
  - 对多轮数据集，正确姿势是：**你自己把多轮数据展平成 LLMRouter 所需的 JSONL 格式**，再让 Step3 去打模型/做评测。

- **最省事方案（推荐）**：绕过 Step2a，手工生成 `query_data_train.jsonl` / `query_data_test.jsonl`
  - 把多轮对话的每一轮 turn 转成一条 StandardQueryData（LLMRouter 的 JSONL 行格式），然后在配置里指向它。
  - 配置文件修改位置见：
    - [baseline/LLMRouter/llmrouter/data/sample_config.yaml](baseline/LLMRouter/llmrouter/data/sample_config.yaml)

- **需要额外改造的点**（否则评测不公平/跑不通）：
  - 你需要定义 `task_name`（例如 `coqa`/`quac`/`sharc`/`inscit`）
  - 你需要选择一个 metric：
    - CoQA/QuAC：通常用 **token-level F1 / Exact Match**；LLMRouter 已有 `f1_score`/`exact_match_score`，但它们目前主要在 Step3 里按 task_name 分支调用。
    - ShARC：答案空间更复杂（可能是 yes/no/irrelevant/或 follow-up question）；建议单独写一个 evaluation 分支。
    - INSCIT：如果只评“response 文本”，可以用生成式指标（例如 EM/F1/ROUGE），但若要评 grounded evidence，则需要额外实现。

---

## 4) 如何把多轮数据“展平”为 LLMRouter 可用的 JSONL

LLMRouter Step3 期望读取的 query JSONL 字段（见 [baseline/LLMRouter/llmrouter/data/README.md](baseline/LLMRouter/llmrouter/data/README.md) ）包含：
- `task_name`: 任务名
- `query`: 输入文本
- `ground_truth`: 参考答案
- `metric`: 评测方法名（LLMRouter 内部会按此选择 evaluator）
- `choices`: 多选题可用（否则 null）
- `task_id`: 可选

### 4.1 CoQA/QuAC 的建议展平方式

对话中第 $t$ 轮的 query，建议包含：
- passage/context
- 历史问答（到 $t-1$）
- 当前问题

示例 query 模板（伪格式）：

```
[Context]
{passage}

[Dialogue History]
Q1: ...
A1: ...
...

[Current Question]
Qt: ...
```

`ground_truth`：当前轮的答案文本（CoQA 的 `answers["input_text"][t]`；QuAC 的 `orig_answers["texts"][t]` 或等价字段）。

`metric`：建议先用 `f1` 或 `em`（需要你在 Step3 里为这些 task_name 接上对应 evaluator，或先简单用 exact match）。

### 4.2 ShARC 的建议展平方式

ShARC 的输入里有“规则 snippet + 场景 scenario + history + 当前问题”，可拼成：

```
[Rule Snippet]
{snippet}

[Scenario]
{scenario}

[History]
Q: ... A: ...
...

[Question]
{question}
```

`ground_truth`: `answer`

注意：ShARC 的 `answer` 可能是 yes/no/irrelevant 或 follow-up question（取决于具体版本/设定），建议你先抽样 50 条看看分布，再决定 metric。

### 4.3 INSCIT 的建议展平方式（baseline 友好版本）

如果你只想把 INSCIT 当作“多轮生成任务”来训练/评测路由器（忽略 evidence），可以：
- query = `context`（历史对话） + 当前 user 问
- ground_truth = `labels[*].response`（取一个标注或做多参考）

如果你要用 evidence：那就不是简单替换 baseline 数据集了，需要引入检索/证据选择模块，评测也要更改。

---

## 5) 你现在这套 baseline 中，改哪里最合适？

- 你不必改 Step2a 的数据集列表；直接准备 JSONL 更稳。
- 你需要改的是：
  - 配置文件 `data_path.query_data_train/query_data_test` 指向你生成的 JSONL
  - Step3 的评测脚本 [baseline/LLMRouter/llmrouter/data/api_calling_evaluation.py](baseline/LLMRouter/llmrouter/data/api_calling_evaluation.py) 中对 `task_name/metric` 的分支（让它能评 CoQA/QuAC/ShARC/INSCIT）

如果你只想先“跑通管线拿到 routing_data”，也可以先把 `metric` 都设成一个你已支持的简单指标（比如 exact match），等跑通后再补齐更合理的评测。

---

## 6) 建议的落地步骤（最少改动）

1) 选一个数据集（建议从 CoQA 或 QuAC 开始）
2) 写一个小脚本把它展平成 `default_query_train.jsonl/default_query_test.jsonl`
3) 复制并修改 [baseline/LLMRouter/llmrouter/data/sample_config.yaml](baseline/LLMRouter/llmrouter/data/sample_config.yaml) 指向你的 JSONL
4) 跑 Step3（必要时先减少 workers）：
   - `python llmrouter/data/api_calling_evaluation.py --config <你的配置> --workers 8`
5) 再考虑把 metric 做严谨（F1/EM、多参考答案等）

---

如果你希望我直接把“展平脚本 + 一个可跑的 config”也补上（例如 `baseline/convert_multiturn_to_llmrouter_jsonl.py`），你告诉我：
- 你优先选哪个数据集（CoQA/QuAC/ShARC/INSCIT）
- 你希望评测用 EM 还是 F1（或两者都要）
- 你希望用“每轮都作为一个样本”，还是“只取最后一轮/固定轮数”
