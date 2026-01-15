# Cache-Aware LLM Agent Scheduler (MAS)

This project upgrades a simulated cache-aware scheduler into a real, networked multi-agent system using:

- vLLM agent servers (OpenAI-compatible HTTP)
- FastAPI scheduler service (cache-aware routing)
- LangGraph orchestrator (workflow)

## Components

- `scheduler_service.py`: FastAPI app exposing:
  - `POST /register_agent` — register agent metadata `{agent_id, url, skills}`
  - `POST /schedule` — choose best agent for `{context, required_skills}`
  - `POST /update_cache` — feedback loop to update KV-cache affinity
- `orchestrator.py`: LangGraph workflow with two nodes: scheduler -> agent
- `run.py`: Registers agents and performs two conversation turns.
- `scheduler_mvp.py`: Original simulation MVP (kept for reference).

## Setup

Using uv (recommended):

```bash
# install dependencies from pyproject.toml
uv sync --all-groups
```

## Run services

Terminal A (vLLM agents; replace model with yours):

```bash
vllm serve --model <model_name> --port 8001
vllm serve --model <model_name> --port 8002
```

Terminal B (scheduler service) — use 9000 to avoid conflict with vLLM on 8000/8001:

```bash
uv run uvicorn icemas.scheduler_service:app --host 0.0.0.0 --port 9000
```

Terminal C (orchestrator + demo run):

```bash
uv run python run.py
```

## One-click E2E with mock agents

You can run a full end-to-end demo using lightweight mock OpenAI-compatible agents:

```bash
chmod +x ./e2e.sh
./e2e.sh
```

This script will:

- Start two mock agent servers on ports 8001 and 8002
- Start the scheduler service on port 9000
- Register both agents to the scheduler
- Run the orchestrator for two conversation turns and print the outputs

Notes:

- The orchestrator will call the scheduler to select an agent, then call the chosen vLLM server at `{agent_url}/v1` using an OpenAI-compatible chat API (via LangChain's ChatOpenAI with `base_url`).
- After each agent call, the orchestrator notifies the scheduler via `/update_cache` so the next turn benefits from cache affinity.

## Expected behavior

- Turn 1: Cold start; any agent may be selected (tie-broken deterministically).
- Turn 2: With the same conversation context extended, the selected agent should remain the same due to cache affinity.

## Troubleshooting

- If `ImportError` occurs, ensure you installed the `requirements.txt` in your active Python environment.
- If vLLM requires an API key, set `OPENAI_API_KEY` or adapt `ChatOpenAI` construction accordingly.
- If your vLLM uses a different path than `/v1`, update `orchestrator.py`.

## Run predictor

Run simulated vllm server in terminal session 1:

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run uvicorn immas.sim.app:app --reload
```

Run client in terminal session 2:

```bash
export HF_ENDPOINT=https://hf-mirror.com
MAX_DIALOGUES=64 MAX_TURNS=5 MAX_CONCURRENCY=32 VERBOSE=1 RUN_LOG_PATH=coqa_run.jsonl uv run -m immas.client.runners.coqa_runner
```

Analyze data in terminal session 3:

```bash
uv run -m immas.analysis.run_analyzer --log coqa_run.jsonl --outdir run_analysis
```
