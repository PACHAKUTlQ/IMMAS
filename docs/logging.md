# JSONL log format (router_run.jsonl)

Module: `immas.router.logger`

The router writes one JSON object per request.

## Record schema: `RouterLogRecord`

### Identifiers and timing

- `run_id`: string, propagated from header
- `t_start_monotonic`, `t_end_monotonic`: router monotonic timestamps
- `backend_id`, `backend_base_url_v1`
- `model`, `source`, `dialogue_id`, `turn_number`

### Router decision-time features

- `prompt_chars`: length of serialized prompt
- `cached_prompt_chars`: cached serialized prompt length for that key
- `kvmatch_lcp_chars`: LCP chars with cached prompt
- `kvmatch_text`: $\in [0,1]$ LCP ratio proxy
- `router_inflight`: inflight requests at time of routing
- `router_rps_1s`: approximate RPS over last 1 second

### Predictions (online model outputs)

- `pred_latency_ms`
- `pred_cost_tokens`
- `pred_perf_prob`
- `pred_cache_ratio`

### Observations (post-backend)

- `completion_id`: `resp_json["id"]` if available
- `obs_latency_ms`: router-measured backend forward time (E2E inside router forwarding section)
- `obs_prompt_tokens`
- `obs_completion_tokens`
- `obs_total_tokens`
- `obs_cached_tokens`
- `obs_cache_ratio`
- `correct`: currently always `true` (placeholder)
- `error`: optional string, set if non-2xx backend status

## Interpreting cache metrics

- `kvmatch_text` is router-side **textual prefix reuse** proxy.
- `obs_cached_tokens` is backend-reported (if available), and is used as supervision for the cache calibrator.

You typically want to validate that:

- higher `kvmatch_text` correlates with higher `obs_cache_ratio`
- increasing turn number increases `kvmatch_text` within a dialogue (after the first successful response)
