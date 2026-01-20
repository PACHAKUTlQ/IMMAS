# Router design and behavior

## Purpose

The router is a FastAPI service that implements:

- `GET /v1/models` (passthrough to backend)
- `POST /v1/chat/completions` (feature extraction + prediction + forward + logging + online update)

It is designed to be “OpenAI-compatible enough” for client libraries like `openai.AsyncOpenAI`.

## Endpoints

### `GET /v1/models`

- Calls backend `/v1/models`
- Returns backend status code and JSON

### `POST /v1/chat/completions`

Behavior:

1. Parse request JSON; expects at least:
   - `model`
   - `messages`
2. Serialize messages deterministically with `serialize_chat_messages(messages)`
3. Compute prefix-cache match feature:
   - `kvmatch_text`
   - `kvmatch_lcp_chars`
   - `cached_prompt_chars`
4. Track router load in a sliding $1$ second window:
   - `router_inflight`
   - `router_rps_1s`
5. Run predictor:
   - predicted latency, cost, performance probability, cache ratio
6. Forward request to backend `/v1/chat/completions`
7. Parse response usage robustly via `parse_usage`
8. Update predictor (online learning) and prefix cache on success
9. Write JSONL log record
10. Return backend response JSON and status (transparent)

## Deterministic prompt serialization

Module: `immas.openai.chat.serialize_chat_messages`

### Key property (prefix consistency)

If message list $M_t$ is an element-wise prefix of $M_{t+1}$, then:

$$
\text{serialize}(M_t) \text{ is a byte-for-byte prefix of } \text{serialize}(M_{t+1})
$$

This enables cheap **longest common prefix** computations that reflect append-only multi-turn chat.

### Serialization format (conceptual)

For each message:

- Emit a role header like `<<role:user>>\n` (or includes name)
- Emit content (string or deterministic JSON encoding)
- Emit terminator `\n<<end>>\n`

## Prefix cache (KV match proxy)

Module: `immas.router.prefix_cache.TextPrefixCache`

Keyed by:

- `(backend_id, model, dialogue_id)`

Stored value:

- last known serialized prompt text (including assistant messages appended by router)

Matching:

- Compute `lcp_chars = common_prefix_length(cached_text, prompt_text)`
- Compute `ratio = lcp_chars / len(prompt_text)` clamped to $[0, 1]$

Update policy:

- Only update on successful backend calls (status $2xx$)
- Requires extracting an assistant message from the backend response; if absent (e.g., tool calls), cache may not update.

## Load tracking

Module: `immas.common.load.AsyncLoadTracker`

Tracks:

- inflight requests (exact)
- approximate RPS over a sliding window of length `window_s` (default $1$ second)

RPS estimation:

- stores start timestamps in a deque
- purges timestamps older than `now - window_s`
- computes:

$$
\text{rps} = \frac{\#\text{starts in window}}{\text{window\_s}}
$$

Used as prediction-time features:

- `router_inflight`
- `router_rps_1s`

## Online predictor

Module: `immas.router.predictor.AgentPredictor`

### Goals

Predict, at routing time:

- latency (ms)
- cost (tokens)
- performance (currently placeholder label always `True`)
- cache reuse ratio (calibration model)

### Leakage avoidance

- The router **does not** use observed cached tokens as a prediction-time feature.
- It **does** train a model to predict cache ratio from router-known features.
- The latency/cost models consume **predicted** cache ratio (`pred_cache_ratio`), not the real one.

### Feature sets

Base features:

- `model`, `source`, `turn_number`
- `prompt_chars` (length of serialized prompt)
- `kvmatch_text`
- `router_inflight`, `router_rps_1s`

Cache calibrator predicts:

- `pred_cache_ratio` in $[0,1]$

Full features for latency/cost/perf:

- base features + `pred_cache_ratio`

### Model implementation

Uses River pipelines:

- OneHotEncoder + HoeffdingTreeRegressor for latency
- OneHotEncoder + HoeffdingTreeRegressor for cost
- OneHotEncoder + HoeffdingTreeClassifier for performance
- OneHotEncoder + HoeffdingTreeRegressor for cache ratio

## Token usage parsing

Module: `immas.openai.usage.parse_usage`

Supports multiple schemas:

- Chat Completions usage:
  - `prompt_tokens`, `completion_tokens`, `total_tokens`
  - optional `prompt_tokens_details.cached_tokens`
- Responses/Realtimes usage:
  - `input_tokens`, `output_tokens`, `total_tokens`
  - optional `input_token_details.cached_tokens`

Cache ratio label:

$$
\text{obs\_cache\_ratio} =
\begin{cases}
\frac{\text{cached\_tokens}}{\text{prompt\_tokens}}, & \text{if prompt\_tokens} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

## Header propagation

Inbound headers used by router (from the loadgen):

- `X-IMMAS-RUN-ID`
- `X-IMMAS-DIALOGUE-ID`
- `X-IMMAS-TURN-NUMBER`
- `X-IMMAS-SOURCE`

Passthrough headers forwarded to backend (safe subset):

- `Authorization` (lowercased in code list)

The router also injects the IMMAS headers into backend request headers.
