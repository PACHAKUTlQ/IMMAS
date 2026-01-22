# CoQA dialogue load generator

Module: `immas.client.runners.coqa_loadgen`

## Purpose

Generate multi-turn chat traffic from the CoQA dataset:

- Concurrency is at the **dialogue** level (multiple dialogues run concurrently).
- Turns within a single dialogue are **sequential** (turn $t+1$ depends on assistant output from turn $t$).

The loadgen:

- calls the router’s `/v1/chat/completions`
- does not perform caching
- does not log JSONL (router handles logging)

## How a dialogue is constructed

For each dialogue:

1. Create stable initial messages:
   - system instruction
   - user message containing the story

2. For each turn:
   - append a new user question message
   - call chat completions
   - append assistant response verbatim to messages (no stripping)

This makes the prompt history append-only and stable, which is essential for prefix matching.

## Concurrency model

- One asyncio task per dialogue.
- A global semaphore `send_sem` caps the number of in-flight requests to `MAX_CONCURRENCY`.

This yields:

- dialogue-level parallelism
- bounded total concurrency

## Progress and stats

- A progress bar tracks total number of requests:
  - `sum(min(d.num_turns(), MAX_TURNS) for d in dialogues)`

- GlobalStats:
  - `n_requests`, `n_errors`
  - `total_latency_ms`
  - `total_tokens` (from OpenAI client response usage if available)

Summary prints averages:

$$
\text{avg\_latency\_ms} = \frac{\text{total\_latency\_ms}}{\text{n\_requests}}
\quad,\quad
\text{avg\_tokens} = \frac{\text{total\_tokens}}{\text{n\_requests}}
$$

## CoQA loading

Uses HuggingFace datasets:

- Dataset name: `stanfordnlp/coqa`
- Split: `COQA_SPLIT` (default `validation`)

Dialogue identity:

- CoQA does not always supply an explicit id field.
- The loader derives a stable synthetic id:

$$
\text{dialogue\_id} = \text{"coqa\_sha1\_"} + \text{sha1}(\text{story} + \text{questions})
$$

This id is propagated in router headers so per-dialogue cache keys and logs remain consistent.
