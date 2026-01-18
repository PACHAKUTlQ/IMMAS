# Predictor Framework Documentation

> [!warning]
>
> This documentation is outdated.

## Overview

The **Predictor** framework is a simulation and experimentation environment designed to model, measure, and predict the performance characteristics (latency, cost, and correctness) of Large Language Model (LLM) serving systems.

It consists of three main components:

1. **Online Predictor (`core`):** An incremental learning system using `river` to predict request latency and cost in real-time based on input features (prompt length, cache hits, system load).
2. **Simulation Server (`sim`):** A FastAPI-based mock server that mimics OpenAI's API. It serves ground-truth answers from the CoQA dataset while simulating complex latency physics (TTFT, decode time, queueing, and stalls) characteristic of engines like vLLM.
3. **Client Runner (`client`):** A high-concurrency load generator that runs dialogues, queries the predictor, executes requests against the server, and updates the predictor with observed ground truths.

---

## Directory Structure

```text
predictor/
├── analysis/           # Post-run analysis and visualization
│   └── run_analyzer.py
├── client/             # Load generation and logging
│   ├── logger.py       # Async JSONL logging
│   └── runners/        # Experiment runners
│       └── coqa_runner.py
├── core/               # Machine Learning logic
│   ├── engine.py       # River-based AgentPredictor
│   └── prefix_cache.py # KV-cache feature engineering
├── data/               # Dataset handling
│   └── coqa/           # Stanford CoQA specific logic
│       ├── loader.py
│       └── prompt.py
└── sim/                # Fake Server & Latency Physics
    ├── app.py          # FastAPI entrypoint
    ├── latency.py      # vLLM-like latency mathematics
    ├── load.py         # Server-side concurrency tracking
    └── traces.py       # Debug trace storage
```

---

## 1. Core Engine (`predictor.core`)

The core module implements the client-side predictive intelligence. It uses **Online Machine Learning** to adapt to changing server conditions without a separate training phase.

### `engine.py`: AgentPredictor

The `AgentPredictor` orchestrates feature extraction and model inference.

* **Models:**
  * **Latency & Cost:** Uses `tree.HoeffdingTreeRegressor` (via `river`). This tree handles non-stationary data and learns incrementally.
  * **Performance (Correctness):** Uses `tree.HoeffdingTreeClassifier`.
* **Pipeline:**
    1. **Feature Construction:** `make_features` combines static inputs (prompt chars) with dynamic state (client-side inflight count, estimated KV-cache match).
    2. **Prediction:** Returns a tuple of `(value, std_dev)`.
    3. **Update:** The `update()` method feeds observed real-world values back into the Hoeffding trees to refine the model.

### `prefix_cache.py`: TextPrefixCache

A utility to estimate **KV-Cache Reuse**.

* It tracks the last known text state of a (model, dialogue) pair.
* Calculates the **Match Ratio**: $$\frac{\text{Length of Common Prefix}}{\text{Total Prompt Length}}$$
* This ratio is a critical feature for the predictor, as higher cache hits significantly lower Time To First Token (TTFT).

---

## 2. Simulation Server (`predictor.sim`)

The simulation server mimics an OpenAI-compatible endpoint (`/v1/chat/completions`) but uses a mathematical model to generate latency rather than GPU inference.

### `app.py`

The FastAPI entry point.

* **Oracle:** Loads the CoQA dataset. When a prompt comes in, it parses the `dialogue_id` and `turn_number` to look up the "Gold" answer instantly.
* **Trace Store:** Stores detailed internal metrics (utilization, simulated stall time) keyed by `completion_id` so the client can retrieve them for analysis.

### `latency.py`: VllmLatencySimulator

The physics engine for latency. It decomposes latency into specific components based on token counts ($N_{prompt}$, $N_{gen}$) and server load ($u$).

**Key Latency Components:**

1. **Base TTFT ($T_{ttft}$):** Modeled as fixed overhead + prefill time.
    $$T_{ttft} = T_{overhead} + \frac{N_{prompt}}{TPS_{prefill}}$$
2. **Decode Time ($T_{decode}$):** Linear generation time.
    $$T_{decode} = \frac{N_{gen}}{TPS_{decode}}$$
3. **Load Scaling:** Both TTFT and Decode are scaled by utilization $u$ (current inflight / capacity).
    $$\text{Multiplier} = 1 + \alpha \cdot u^\beta$$
4. **Queueing:** Applies when $u > 1.0$.
5. **Stochastic Stalls:** To mimic "batch" pauses or GC pauses in vLLM, the simulator probabilistically injects "stalls" when utilization is high ($u > 0.85$).

### `load.py`: AsyncLoadTracker

Tracks the server-side concurrency.

* Maintains a sliding window to calculate Requests Per Second (RPS).
* Provides the $u$ (utilization) variable used by the latency simulator.

---

## 3. Client & Runner (`predictor.client`)

### `runners/coqa_runner.py`

The main experiment driver.

* **Concurrency:** Uses `asyncio.Semaphore` to limit global concurrency.
* **Dialogue Management:** Runs multiple dialogues in parallel, but turns *within* a dialogue are sequential (Turn 2 waits for Turn 1).
* **The Loop:**
    1. **Predict:** Ask `AgentPredictor` for estimated latency/cost.
    2. **Execute:** Call `POST /v1/chat/completions`.
    3. **Fetch Debug:** (Optional) Call `GET /v1/internal/chat_completions/{id}` to get ground truth server stats.
    4. **Update:** Teach the `AgentPredictor` with the actual results.
    5. **Log:** Write a JSONL record.

### `logger.py`

A non-blocking, structured JSONL logger. It uses a background `asyncio` task to write to disk, ensuring that high-throughput logging does not block the request loop.

---

## 4. Data Handling (`predictor.data`)

### CoQA Loader

* Loads `stanfordnlp/coqa` from HuggingFace.
* **Deterministic ID:** If the dataset lacks IDs, it generates a stable SHA1 hash from the story content to ensure consistency between client and server.

### Prompt Formatting

* Constructs prompts with specific markers (`### COQA_DIALOGUE_ID`, `### CONVERSATION`, `Q{n}:`).
* This strict formatting allows the "Fake Server" to parse the prompt text back into structured data to look up the correct answer.

---

## 5. Analysis (`predictor.analysis`)

### `run_analyzer.py`

Parses the `jsonl` logs generated by the client to evaluate the predictor's performance.

**Metrics:**

* **MAE / RMSE:** For Latency (ms) and Cost (tokens).
* **Correlation:** Pearson correlation between Predicted vs. Observed.
* **Accuracy:** Percentage of correct answers (sanity check for the simulator).

**Plots:**

* `latency_timeseries.png`: Observed vs. Predicted over time.
* `latency_scatter.png`: Correlation visualization.
* `util_vs_latency.png`: Highlights how latency degrades as server utilization increases, coloring outliers (stalls) in red.

---

## Configuration Reference

The system is heavily driven by Environment Variables.

### Client Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_DIALOGUES` | `3` | Number of CoQA stories to process. |
| `MAX_TURNS` | `5` | Max turns per story. |
| `MAX_CONCURRENCY` | `8` | Max simultaneous requests (Client side). |
| `RUN_LOG_PATH` | `coqa_run.jsonl` | Output path for logs. |
| `VERBOSE` | `0` | If 1, prints one line per request to stdout. |
| `FETCH_SERVER_DEBUG`| `1` | If 1, fetches internal traces from server. |

### Server Simulation Configuration (`predictor.sim.latency`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FAKE_VLLM_CAPACITY` | `32` | Max concurrent requests before queueing starts ($u=1.0$). |
| `FAKE_VLLM_PREFILL_TPS` | `60000` | Tokens/sec for prompt processing. |
| `FAKE_VLLM_DECODE_TPS` | `2500` | Tokens/sec for generation. |
| `FAKE_VLLM_STALL_LOAD_THRESHOLD` | `0.85` | Utilization at which stochastic stalls begin. |
| `FAKE_VLLM_JITTER_STD_S` | `0.002` | Gaussian noise added to latency. |

---

## Usage Guide

### 1. Start the Server

The server must be running to handle requests and provide the "Oracle" answers.

```bash
# Optional: Set HF mirror if needed
export HF_ENDPOINT=https://hf-mirror.com

# Run with uvicorn (auto-reload useful for dev)
uv run uvicorn predictor.sim.app:app --reload --port 8000
```

### 2. Run the Client Experiment

The client loads the dataset, sends requests, and trains the predictor online.

```bash
export MAX_DIALOGUES=64
export MAX_TURNS=5
export MAX_CONCURRENCY=32
export VERBOSE=1
export RUN_LOG_PATH=experiment_logs.jsonl

uv run -m predictor.client.runners.coqa_runner
```

### 3. Analyze Results

Generate metrics and plots from the logs.

```bash
uv run -m predictor.analysis.run_analyzer \
    --log experiment_logs.jsonl \
    --outdir analysis_results
```

*Check `analysis_results/` for PNG plots comparing predicted vs. actual latency.*
