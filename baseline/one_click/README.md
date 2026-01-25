# One-click Online Experiment (many requests → many LLMs)

Entry script:
- [baseline/run_one_click_experiment.sh](../run_one_click_experiment.sh)

## Quick Start

1) Edit the config:
- [baseline/one_click/configs/example_poisson.yaml](configs/example_poisson.yaml)

2) Run:

```bash
bash baseline/run_one_click_experiment.sh baseline/one_click/configs/example_poisson.yaml
```

Artifacts are written to:
- `baseline/logs/<run_id>/`

## Build Multiturn Data (Optional)

Convert CoQA/QuAC into JSONL required by the online experiment (one request per line, including user+assistant history).

Note: If your config references
- `baseline/one_click/data/coqa_<split>_multiturn.jsonl` or
- `baseline/one_click/data/quac_<split>_multiturn.jsonl`

and the file does not exist, the one-click script will download and build it at runtime from HuggingFace (requires `datasets`).

Default behavior (when you do not set limits):
- `MAX_CONVERSATIONS=0` downloads and builds the full dataset
- `MAX_TURNS_PER_CONV=0` keeps all turns per conversation
- Both `train` and `validation` files are generated

For a small smoke run:
- `MAX_CONVERSATIONS=50 MAX_TURNS_PER_CONV=5`

```bash
python baseline/one_click/data/build_multiturn_jsonl.py --dataset coqa --split train \
  --out baseline/one_click/data/coqa_train_multiturn.jsonl --max-conversations 50 --max-turns-per-conv 5 --history-turns all
```

Then point `request_stream.datasets[*].path` to the generated JSONL.

## Directory Overview

- `online_experiment.py`: online request stream + batching + router + (optional shadow compare) + artifacts
- `vllm_launch.py`: launch multiple vLLM OpenAI-compatible servers (multi-port, auto GPU selection)
- `wait_for_vllm.py`: health check (`/v1/models`)
- `data/`: data build scripts and sample data (extensible)
- `llm_candidates/`: candidate model JSON for LLMRouter configs
- `router_configs/`: LLMRouter YAML configs (referencing the candidate JSON)

## Important Flags

- `skip_vllm: true`: do not launch vLLM; only run the router (assumes endpoints are already up)
- `dry_run: true`: do not call any LLM; only run arrival/batching/logging (pipeline validation)
- `shadow_compare.enabled: true`: call additional candidates as shadow runs for comparison
