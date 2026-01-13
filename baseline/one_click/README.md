# One-click 在线实验（many requests → many LLMs）

入口脚本：
- [baseline/run_one_click_experiment.sh](../run_one_click_experiment.sh)

## 快速开始

1) 修改配置：
- [baseline/one_click/configs/example_poisson.yaml](configs/example_poisson.yaml)

2) 一键运行：

```bash
bash baseline/run_one_click_experiment.sh baseline/one_click/configs/example_poisson.yaml
```

产物会写到：
- `baseline/logs/<run_id>/`

## 生成多轮数据（可选）

把 CoQA/QuAC 转成在线实验需要的 JSONL（每行一个 request，包含 user+assistant 历史）：

说明：如果你的配置里直接引用了
- `baseline/one_click/data/coqa_<split>_multiturn.jsonl` 或
- `baseline/one_click/data/quac_<split>_multiturn.jsonl`

且文件不存在，那么一键脚本会在运行时自动从 HuggingFace 下载并生成（需要安装 `datasets`）。

默认行为（不手动设置上限时）：
- `MAX_CONVERSATIONS=0` 表示下载并生成全量对话
- `MAX_TURNS_PER_CONV=0` 表示每个对话保留全量 turns
- 会分别生成 `train` 和 `validation` 两个文件（分开落盘）

如果只想做小规模 smoke：
- `MAX_CONVERSATIONS=50 MAX_TURNS_PER_CONV=5`

```bash
python baseline/one_click/data/build_multiturn_jsonl.py --dataset coqa --split train \
	--out baseline/one_click/data/coqa_train_multiturn.jsonl --max-conversations 50 --max-turns-per-conv 5 --history-turns all
```

然后在配置里把 `request_stream.datasets[*].path` 指向生成的 JSONL。

## 目录说明

- `online_experiment.py`：在线请求流 + batching + router +（可选 shadow compare）+ 产物落盘
- `vllm_launch.py`：按配置启动多个 vLLM openai-compatible server（多端口，多 GPU 自动选择）
- `wait_for_vllm.py`：健康检查（`/v1/models`）
- `data/`：数据构建脚本与示例数据（可扩展）
- `llm_candidates/`：给 LLMRouter router config 使用的候选模型 JSON
- `router_configs/`：给 LLMRouter 加载 router 的 YAML（引用上面的候选模型 JSON）

## 重要开关

- `skip_vllm: true`：不启动 vLLM，只跑路由（假设 endpoints 已运行）
- `dry_run: true`：不调用任何 LLM，仅跑 arrival/batching/logging（用于验证管线与产物）
- `shadow_compare.enabled: true`：除了 router 选择的模型，还会对其它候选模型做 shadow 调用，保留候选对比结果
