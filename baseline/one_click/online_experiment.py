#!/usr/bin/env python3
"""Online many-requests-many-LLMs experiment runner (baseline).

Implements:
- Poisson arrival process
- Batching: flush on size, otherwise flush on time window
- Baseline many-to-many by concatenating one-to-many per request
- Per-model concurrency limits
- Optional shadow comparison calls for all candidates
- Rich artifacts under a run directory

All outputs stay under baseline/.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency PyYAML. Install pyyaml.") from e


# -----------------------------------------------------------------------------
# Lazy-import LLMRouter (heavy) inside run_experiment()
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
BASELINE_DIR = THIS_DIR.parent

_load_router = None
_call_api = None
_calculate_task_performance = None


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    api_endpoint: str
    api_name: str
    service: Optional[str] = None
    per_model_max_concurrency: int = 4


@dataclass
class RequestItem:
    request_id: str
    dataset: str
    conversation_id: Optional[str]
    turn_id: Optional[int]
    messages: List[Dict[str, str]]
    ground_truth: Optional[Any] = None
    metric: Optional[str] = None
    task_name: Optional[str] = None


@dataclass
class RoutingDecision:
    request_id: str
    batch_id: int
    chosen_model: str
    router_latency_ms: float
    routing_result: Dict[str, Any]


@dataclass
class ModelCall:
    request_id: str
    batch_id: int
    model_name: str
    api_endpoint: str
    api_name: str
    is_chosen: bool
    success: bool
    llm_latency_ms: float
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None
    task_performance: Optional[float] = None
    social_welfare: Optional[float] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def now_ms() -> float:
    return time.time() * 1000.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_messages_as_prompt(messages: List[Dict[str, str]]) -> str:
    """Collapse chat messages into a single user prompt string.

    LLMRouter's call_api currently sends a single user message, so we serialize
    the conversation into plain text.
    """
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"[System]\n{content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    # Ensure it ends with assistant turn for generation
    if not lines or not lines[-1].startswith("User:"):
        return "\n".join(lines)
    return "\n".join(lines) + "\nAssistant:"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


# -----------------------------------------------------------------------------
# Dataset sampling
# -----------------------------------------------------------------------------

class JsonlDatasetSampler:
    def __init__(
        self,
        name: str,
        path: Path,
        mode: str = "sequential",
        metric_override: Optional[str] = None,
        task_name_override: Optional[str] = None,
        dataset_override: Optional[str] = None,
    ):
        self.name = name
        self.path = path
        self.mode = mode
        self.metric_override = metric_override
        self.task_name_override = task_name_override
        self.dataset_override = dataset_override
        self.rows = load_jsonl(path)
        if not self.rows:
            raise ValueError(f"Dataset is empty: {path}")
        self._idx = 0

    def next_request(self) -> RequestItem:
        if self.mode == "random":
            row = random.choice(self.rows)
        else:
            row = self.rows[self._idx % len(self.rows)]
            self._idx += 1

        # Single-turn compatibility: LLMRouter query jsonl usually contains {"query": ...}
        if "messages" not in row:
            query = row.get("query") or row.get("formatted_query") or ""
            messages = [{"role": "user", "content": str(query)}]
        else:
            messages = row.get("messages") or []

        req_id = row.get("request_id") or f"{self.name}-{self._idx}-{int(time.time()*1000)}"
        return RequestItem(
            request_id=str(req_id),
            dataset=str(self.dataset_override or row.get("dataset") or self.name),
            conversation_id=row.get("conversation_id"),
            turn_id=row.get("turn_id"),
            messages=messages,
            ground_truth=row.get("ground_truth") or row.get("answer") or row.get("gt"),
            metric=self.metric_override or row.get("metric"),
            task_name=self.task_name_override or row.get("task_name"),
        )


class WeightedDatasetMixer:
    def __init__(self, samplers: List[Tuple[JsonlDatasetSampler, float]]):
        if not samplers:
            raise ValueError("No datasets configured")
        self.samplers = samplers
        total = sum(w for _, w in samplers)
        if total <= 0:
            raise ValueError("Sum of dataset weights must be > 0")
        # Normalize
        self.weights = [w / total for _, w in samplers]

    def next_request(self) -> RequestItem:
        sampler = random.choices([s for s, _ in self.samplers], weights=self.weights, k=1)[0]
        return sampler.next_request()


# -----------------------------------------------------------------------------
# Online engine
# -----------------------------------------------------------------------------

class PerModelLimiter:
    def __init__(self, limits: Dict[str, int]):
        self.semaphores: Dict[str, asyncio.Semaphore] = {
            model_name: asyncio.Semaphore(max(1, int(limit))) for model_name, limit in limits.items()
        }

    def sem(self, model_name: str) -> asyncio.Semaphore:
        return self.semaphores.get(model_name) or asyncio.Semaphore(1)


async def call_llm_with_limit(
    limiter: PerModelLimiter,
    candidate: Candidate,
    prompt: str,
    timeout_sec: int,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    async with limiter.sem(candidate.name):
        if dry_run:
            await asyncio.sleep(0.01)
            return {
                "response": "[dry-run]",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "token_num": 0,
                "response_time": 0.01,
            }

        req: Dict[str, Any] = {
            "api_endpoint": candidate.api_endpoint,
            "query": prompt,
            "model_name": candidate.name,
            "api_name": candidate.api_name,
        }
        if candidate.service:
            req["service"] = candidate.service
        if system_prompt:
            req["system_prompt"] = system_prompt

        # call_api is sync; run in a thread
        if _call_api is None:
            raise RuntimeError("LLMRouter call_api not initialized")

        result = await asyncio.to_thread(
            _call_api,
            req,
            None,
            max_tokens,
            temperature,
            0.9,
            timeout_sec,
            3,
        )
        return result


def build_candidates(cfg: Dict[str, Any], run_logger: logging.Logger) -> Dict[str, Candidate]:
    candidates_path = Path(cfg["llm_candidates_path"]).resolve()
    if not candidates_path.exists():
        raise FileNotFoundError(f"llm_candidates_path not found: {candidates_path}")
    data = json.loads(candidates_path.read_text(encoding="utf-8"))

    # Optional: per-model concurrency specified in vllm section
    per_model_limits: Dict[str, int] = {}
    for m in (cfg.get("vllm") or {}).get("models", []) or []:
        name = m.get("name")
        if name:
            per_model_limits[str(name)] = int(m.get("per_model_max_concurrency") or 4)

    out: Dict[str, Candidate] = {}
    for name, meta in data.items():
        out[name] = Candidate(
            name=name,
            api_endpoint=str(meta["api_endpoint"]),
            api_name=str(meta.get("model") or name),
            service=meta.get("service"),
            per_model_max_concurrency=int(per_model_limits.get(name, 4)),
        )

    run_logger.info("Loaded %d candidates from %s", len(out), candidates_path)
    return out


def setup_logging(run_dir: Path) -> logging.Logger:
    ensure_dir(run_dir)
    logger = logging.getLogger("one_click")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


async def poisson_arrivals(qps: float):
    # exponential inter-arrival times with mean 1/qps
    while True:
        if qps <= 0:
            await asyncio.sleep(1)
            continue
        dt = random.expovariate(qps)
        await asyncio.sleep(dt)
        yield None


async def run_experiment(cfg: Dict[str, Any], run_dir: Path) -> None:
    logger = setup_logging(run_dir)
    logger.info("Run dir: %s", run_dir)

    # LLMRouter's LiteLLM calling path requires API_KEYS to be set.
    # For local vLLM OpenAI-compatible endpoints, we can safely default it to OPENAI_API_KEY.
    if not os.environ.get("API_KEYS"):
        fallback_key = os.environ.get("OPENAI_API_KEY") or "local"
        os.environ["API_KEYS"] = fallback_key
        logger.info("API_KEYS not set; defaulting API_KEYS from OPENAI_API_KEY (len=%d)", len(fallback_key))

    # Lazy import LLMRouter after we have logging
    global _load_router, _call_api, _calculate_task_performance
    if _load_router is None or _call_api is None or _calculate_task_performance is None:
        logger.info("Importing LLMRouter (may take a while on first run)...")
        llmrouter_dir = BASELINE_DIR / "LLMRouter"
        if str(llmrouter_dir) not in sys.path:
            sys.path.insert(0, str(llmrouter_dir))
        from llmrouter.cli.router_inference import load_router as _lr  # noqa: WPS433
        from llmrouter.utils.api_calling import call_api as _ca  # noqa: WPS433
        from llmrouter.utils.evaluation import calculate_task_performance as _ctp  # noqa: WPS433

        _load_router = _lr
        _call_api = _ca
        _calculate_task_performance = _ctp
        logger.info("LLMRouter imported")

    dry_run = bool(cfg.get("dry_run", False))

    # Load router
    router_cfg = cfg["router"]
    router_name = str(router_cfg["name"])
    router_config_path = str(router_cfg["config_path"])
    load_model_path = router_cfg.get("load_model_path")

    logger.info("Loading router=%s config=%s", router_name, router_config_path)
    router = _load_router(router_name, router_config_path, load_model_path)
    logger.info("Router loaded: %s", type(router).__name__)

    candidates = build_candidates(cfg, logger)
    limiter = PerModelLimiter({name: c.per_model_max_concurrency for name, c in candidates.items()})

    # Datasets
    samplers: List[Tuple[JsonlDatasetSampler, float]] = []
    for ds in cfg["request_stream"]["datasets"]:
        if ds.get("type") != "jsonl":
            raise ValueError("Only type=jsonl is supported in v0")
        ds_path = Path(ds["path"]).resolve()
        metric_override = ds.get("metric_override")
        task_name_override = ds.get("task_name_override")
        dataset_override = ds.get("dataset_override")
        samplers.append(
            (
                JsonlDatasetSampler(
                    ds["name"],
                    ds_path,
                    mode=str(ds.get("sampling") or "sequential"),
                    metric_override=metric_override,
                    task_name_override=task_name_override,
                    dataset_override=dataset_override,
                ),
                float(ds.get("weight", 1.0)),
            )
        )

    mixer = WeightedDatasetMixer(samplers)

    # Outputs
    figures_dir = run_dir / "figures"
    ensure_dir(figures_dir)

    write_requests = bool(cfg.get("output", {}).get("write_requests_jsonl", True))
    write_calls = bool(cfg.get("output", {}).get("write_model_calls_jsonl", True))
    write_decisions = bool(cfg.get("output", {}).get("write_routing_decisions_jsonl", True))
    write_metrics_csv = bool(cfg.get("output", {}).get("write_metrics_csv", True))

    req_f = (run_dir / "requests.jsonl").open("w", encoding="utf-8") if write_requests else None
    calls_f = (run_dir / "model_calls.jsonl").open("w", encoding="utf-8") if write_calls else None
    dec_f = (run_dir / "routing_decisions.jsonl").open("w", encoding="utf-8") if write_decisions else None
    metrics_f = (run_dir / "metrics.csv").open("w", encoding="utf-8") if write_metrics_csv else None

    if metrics_f:
        metrics_f.write(
            "batch_id,batch_size,flush_reason,"
            "batch_duration_ms,queue_wait_p50_ms,router_p50_ms,chosen_llm_p50_ms,"
            "chosen_success_rate,shadow_calls,errors,"
            "chosen_task_perf_p50,chosen_total_tokens_p50,chosen_social_welfare_p50\n"
        )

    # Batching config
    batching = cfg["batching"]
    max_batch_size = int(batching["max_batch_size"])
    time_window_ms = int(batching["time_window_ms"])

    execution = cfg["execution"]
    timeout_sec = int(execution.get("timeout_sec", 60))
    max_tokens = int(execution.get("max_tokens", 512))
    temperature = float(execution.get("temperature", 0.2))

    shadow = cfg.get("shadow_compare") or {}
    shadow_enabled = bool(shadow.get("enabled", False))

    social_welfare_cfg = cfg.get("social_welfare") or {}
    token_cost_weight = 1e-3*float(social_welfare_cfg.get("token_cost_weight", 1.0))
    token_cost_field = str(social_welfare_cfg.get("token_cost_field", "total_tokens")).strip()

    # Arrival config
    stream_cfg = cfg["request_stream"]
    qps = float(stream_cfg.get("poisson_qps", 1.0))
    max_requests = int(stream_cfg.get("max_requests", 1000))
    max_duration_sec = int(stream_cfg.get("max_duration_sec", 3600))

    # Metrics accumulators
    e2e_latencies: List[float] = []
    router_latencies: List[float] = []
    chosen_llm_latencies: List[float] = []
    error_count = 0
    chosen_error_count = 0
    shadow_error_count = 0
    total_calls = 0
    chosen_calls = 0
    per_model_chosen: Dict[str, int] = {}
    chosen_task_perfs: List[float] = []
    per_model_chosen_task_perfs: Dict[str, List[float]] = {}
    chosen_total_tokens: List[int] = []
    per_model_chosen_total_tokens: Dict[str, List[int]] = {}
    chosen_social_welfares: List[float] = []

    start_t = time.time()
    buffer: List[Tuple[RequestItem, float]] = []  # (req, enqueue_ms)
    batch_id = 0

    # Shadow tasks are not part of E2E; await them at the end for completeness.
    pending_shadow_tasks: List[asyncio.Task] = []

    async def flush(reason: str) -> None:
        nonlocal batch_id, buffer, error_count, total_calls, chosen_calls
        if not buffer:
            return
        batch_id += 1
        local_batch = buffer
        buffer = []

        batch_start_ms = now_ms()

        logger.info("Flush batch=%d size=%d reason=%s", batch_id, len(local_batch), reason)

        # Route each request (concurrent)
        batch_total_calls = 0
        batch_chosen_calls = 0
        batch_shadow_calls = 0
        batch_errors = 0
        batch_chosen_errors = 0
        batch_shadow_errors = 0

        async def handle_one(req: RequestItem, enqueue_ms: float) -> None:
            nonlocal error_count, chosen_error_count, shadow_error_count
            nonlocal total_calls, chosen_calls
            nonlocal batch_total_calls, batch_chosen_calls, batch_shadow_calls
            nonlocal batch_errors, batch_chosen_errors, batch_shadow_errors

            prompt = format_messages_as_prompt(req.messages)
            if req_f:
                req_f.write(json.dumps(dataclasses.asdict(req), ensure_ascii=False) + "\n")

            t0 = now_ms()
            routing_result = await asyncio.to_thread(router.route_single, {"query": prompt})
            t1 = now_ms()

            chosen_model = (
                routing_result.get("model_name")
                or routing_result.get("predicted_llm")
                or routing_result.get("predicted_llm_name")
            )
            if not chosen_model:
                chosen_model = "__unknown__"

            router_ms = t1 - t0
            decision = RoutingDecision(
                request_id=req.request_id,
                batch_id=batch_id,
                chosen_model=chosen_model,
                router_latency_ms=router_ms,
                routing_result=routing_result,
            )
            router_latencies.append(router_ms)

            if dec_f:
                dec_f.write(json.dumps(dataclasses.asdict(decision), ensure_ascii=False) + "\n")

            # Determine candidates to call
            candidate_names = list(candidates.keys())
            if chosen_model not in candidates:
                logger.warning("Chosen model '%s' not in candidates.json; skipping LLM call", chosen_model)
                return

            async def one_call(model_name: str, is_chosen: bool) -> None:
                nonlocal error_count, chosen_error_count, shadow_error_count
                nonlocal total_calls, chosen_calls
                nonlocal batch_total_calls, batch_chosen_calls, batch_shadow_calls
                nonlocal batch_errors, batch_chosen_errors, batch_shadow_errors
                cand = candidates[model_name]
                call_t0 = now_ms()
                try:
                    result = await call_llm_with_limit(
                        limiter,
                        cand,
                        prompt,
                        timeout_sec,
                        max_tokens,
                        temperature,
                        system_prompt=None,
                        dry_run=dry_run,
                    )
                    call_t1 = now_ms()
                    success = "error" not in result
                    err = result.get("error")
                    response = result.get("response") or ""
                    prompt_tokens = int(result.get("prompt_tokens") or 0)
                    completion_tokens = int(result.get("completion_tokens") or 0)
                    total_tokens = int(result.get("token_num") or (prompt_tokens + completion_tokens))

                    token_cost: float
                    if token_cost_field == "prompt_tokens":
                        token_cost = float(prompt_tokens)
                    elif token_cost_field == "completion_tokens":
                        token_cost = float(completion_tokens)
                    else:
                        token_cost = float(total_tokens)

                    task_perf: Optional[float] = None
                    if req.ground_truth is not None and req.metric is not None and req.metric != "online_only":
                        try:
                            task_perf = _calculate_task_performance(
                                prediction=response,
                                ground_truth=req.ground_truth,
                                task_name=req.task_name,
                                metric=req.metric,
                            )
                        except Exception:
                            task_perf = None

                    social_welfare: Optional[float] = None
                    if task_perf is not None:
                        social_welfare = float(task_perf) - (token_cost_weight * token_cost)

                    mc = ModelCall(
                        request_id=req.request_id,
                        batch_id=batch_id,
                        model_name=model_name,
                        api_endpoint=cand.api_endpoint,
                        api_name=cand.api_name,
                        is_chosen=is_chosen,
                        success=bool(success),
                        llm_latency_ms=(call_t1 - call_t0),
                        response=response,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        error=str(err) if err else None,
                        task_performance=task_perf,
                        social_welfare=social_welfare,
                    )

                    if is_chosen and task_perf is not None:
                        chosen_task_perfs.append(float(task_perf))
                        per_model_chosen_task_perfs.setdefault(model_name, []).append(float(task_perf))

                    if is_chosen:
                        chosen_total_tokens.append(int(total_tokens))
                        per_model_chosen_total_tokens.setdefault(model_name, []).append(int(total_tokens))
                        if social_welfare is not None:
                            chosen_social_welfares.append(float(social_welfare))

                    if calls_f:
                        calls_f.write(json.dumps(dataclasses.asdict(mc), ensure_ascii=False) + "\n")

                    total_calls += 1
                    batch_total_calls += 1
                    if is_chosen:
                        chosen_calls += 1
                        batch_chosen_calls += 1
                        chosen_llm_latencies.append(call_t1 - call_t0)
                        per_model_chosen[model_name] = per_model_chosen.get(model_name, 0) + 1
                    else:
                        batch_shadow_calls += 1

                    if not success:
                        error_count += 1
                        batch_errors += 1
                        if is_chosen:
                            chosen_error_count += 1
                            batch_chosen_errors += 1
                        else:
                            shadow_error_count += 1
                            batch_shadow_errors += 1

                except Exception as e:
                    error_count += 1
                    batch_errors += 1
                    total_calls += 1
                    batch_total_calls += 1
                    if is_chosen:
                        chosen_error_count += 1
                        batch_chosen_errors += 1
                        chosen_calls += 1
                        batch_chosen_calls += 1
                    else:
                        shadow_error_count += 1
                        batch_shadow_errors += 1
                        batch_shadow_calls += 1
                    if calls_f:
                        calls_f.write(
                            json.dumps(
                                {
                                    "request_id": req.request_id,
                                    "batch_id": batch_id,
                                    "model_name": model_name,
                                    "is_chosen": is_chosen,
                                    "success": False,
                                    "error": str(e),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            # Always call chosen on critical path
            chosen_task = asyncio.create_task(one_call(chosen_model, True))
            await chosen_task

            # E2E ends when chosen completes (shadow not included)
            e2e_ms = now_ms() - enqueue_ms
            e2e_latencies.append(e2e_ms)

            # Shadow compare runs in background (still logged)
            if shadow_enabled:
                for m in candidate_names:
                    if m == chosen_model:
                        continue
                    pending_shadow_tasks.append(asyncio.create_task(one_call(m, False)))

        await asyncio.gather(*(handle_one(req, enqueue_ms) for req, enqueue_ms in local_batch))

        # Batch metrics row
        batch_end_ms = now_ms()
        queue_waits = [max(0.0, batch_start_ms - enqueue_ms) for _, enqueue_ms in local_batch]
        batch_duration_ms = batch_end_ms - batch_start_ms
        if metrics_f:
            # Snapshot recent latencies for this flush window is hard without per-request bookkeeping;
            # we log rolling p50 values as an online debug signal.
            router_p50 = percentile(router_latencies[-len(local_batch):], 0.50) if router_latencies else float("nan")
            chosen_llm_p50 = percentile(chosen_llm_latencies[-len(local_batch):], 0.50) if chosen_llm_latencies else float("nan")
            chosen_success_rate = 1.0 - (batch_chosen_errors / max(1, batch_chosen_calls))

            recent_perf = chosen_task_perfs[-batch_chosen_calls:] if batch_chosen_calls > 0 else []
            recent_tokens = chosen_total_tokens[-batch_chosen_calls:] if batch_chosen_calls > 0 else []
            recent_welfare = chosen_social_welfares[-len(recent_perf):] if recent_perf else []
            perf_p50 = percentile([float(x) for x in recent_perf], 0.50) if recent_perf else float("nan")
            tokens_p50 = percentile([float(x) for x in recent_tokens], 0.50) if recent_tokens else float("nan")
            welfare_p50 = percentile([float(x) for x in recent_welfare], 0.50) if recent_welfare else float("nan")
            metrics_f.write(
                f"{batch_id},{len(local_batch)},{reason},"
                f"{batch_duration_ms:.2f},{percentile(queue_waits,0.50):.2f},{router_p50:.2f},{chosen_llm_p50:.2f},"
                f"{chosen_success_rate:.4f},{batch_shadow_calls},{batch_errors},"
                f"{perf_p50:.6f},{tokens_p50:.2f},{welfare_p50:.6f}\n"
            )

    # Main loop
    arrivals = poisson_arrivals(qps)
    produced = 0
    batch_open_ms: Optional[float] = None

    async for _ in arrivals:
        if produced >= max_requests:
            break
        if (time.time() - start_t) >= max_duration_sec:
            break

        produced += 1
        req = mixer.next_request()
        enqueue_t = now_ms()
        buffer.append((req, enqueue_t))
        if batch_open_ms is None:
            batch_open_ms = enqueue_t

        # Flush on size
        if len(buffer) >= max_batch_size:
            await flush("size")
            batch_open_ms = None
            continue

        # Flush on time window since last flush check
        if buffer and batch_open_ms is not None and (now_ms() - batch_open_ms) >= time_window_ms:
            await flush("time")
            batch_open_ms = None

    # final flush
    await flush("final")

    if pending_shadow_tasks:
        logger.info("Awaiting %d shadow compare calls...", len(pending_shadow_tasks))
        await asyncio.gather(*pending_shadow_tasks, return_exceptions=True)

    # Close files
    for f in (req_f, calls_f, dec_f, metrics_f):
        if f:
            f.close()

    # Summary
    summary = {
        "run_id": run_dir.name,
        "dry_run": dry_run,
        "router": router_name,
        "requests": produced,
        "batches": batch_id,
        "total_model_calls": total_calls,
        "chosen_model_calls": chosen_calls,
        "error_count": error_count,
        "chosen_error_count": chosen_error_count,
        "shadow_error_count": shadow_error_count,
        "error_rate": (error_count / max(1, total_calls)),
        "latency_ms": {
            "e2e_p50": percentile(e2e_latencies, 0.50),
            "e2e_p95": percentile(e2e_latencies, 0.95),
            "router_p50": percentile(router_latencies, 0.50),
            "chosen_llm_p50": percentile(chosen_llm_latencies, 0.50),
        },
        "per_model_chosen": per_model_chosen,
    }

    if chosen_task_perfs:
        summary["chosen_task_performance"] = {
            "count": len(chosen_task_perfs),
            "mean": float(statistics.mean(chosen_task_perfs)),
            "p50": float(percentile(chosen_task_perfs, 0.50)),
            "p95": float(percentile(chosen_task_perfs, 0.95)),
        }
        per_model_perf = {}
        for model_name, vals in per_model_chosen_task_perfs.items():
            if not vals:
                continue
            per_model_perf[model_name] = {
                "count": len(vals),
                "mean": float(statistics.mean(vals)),
            }
        if per_model_perf:
            summary["per_model_chosen_task_performance"] = per_model_perf

    if chosen_total_tokens:
        summary["chosen_token_usage"] = {
            "count": len(chosen_total_tokens),
            "mean": float(statistics.mean(chosen_total_tokens)),
            "p50": float(percentile([float(x) for x in chosen_total_tokens], 0.50)),
            "p95": float(percentile([float(x) for x in chosen_total_tokens], 0.95)),
            "token_cost_field": token_cost_field,
            "token_cost_weight": token_cost_weight,
        }
        per_model_tokens = {}
        for model_name, vals in per_model_chosen_total_tokens.items():
            if not vals:
                continue
            per_model_tokens[model_name] = {
                "count": len(vals),
                "mean": float(statistics.mean(vals)),
            }
        if per_model_tokens:
            summary["per_model_chosen_token_usage"] = per_model_tokens

    if chosen_social_welfares:
        summary["chosen_social_welfare"] = {
            "count": len(chosen_social_welfares),
            "total": float(sum(chosen_social_welfares)),
            "mean": float(statistics.mean(chosen_social_welfares)),
            "p50": float(percentile(chosen_social_welfares, 0.50)),
            "p95": float(percentile(chosen_social_welfares, 0.95)),
            "definition": "sum_i (task_performance_i - token_cost_weight * token_cost_i) over chosen calls",
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written: summary.json")

    # Figures (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if e2e_latencies:
            plt.figure(figsize=(8, 4))
            plt.hist(e2e_latencies, bins=40)
            plt.title("E2E latency (ms)")
            plt.xlabel("ms")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures_dir / "e2e_latency_hist.png", dpi=150)
            plt.close()

        if per_model_chosen:
            plt.figure(figsize=(8, 4))
            names = list(per_model_chosen.keys())
            vals = [per_model_chosen[n] for n in names]
            plt.bar(names, vals)
            plt.title("Chosen model counts")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(figures_dir / "chosen_model_counts.png", dpi=150)
            plt.close()

        if chosen_task_perfs:
            plt.figure(figsize=(8, 4))
            plt.hist(chosen_task_perfs, bins=40)
            plt.title("Chosen task performance")
            plt.xlabel("score")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures_dir / "chosen_task_performance_hist.png", dpi=150)
            plt.close()

        if chosen_total_tokens:
            plt.figure(figsize=(8, 4))
            plt.hist(chosen_total_tokens, bins=40)
            plt.title("Chosen token usage")
            plt.xlabel("tokens")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures_dir / "chosen_token_usage_hist.png", dpi=150)
            plt.close()

        if chosen_social_welfares:
            plt.figure(figsize=(8, 4))
            plt.hist(chosen_social_welfares, bins=40)
            plt.title("Chosen social welfare")
            plt.xlabel("welfare")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures_dir / "chosen_social_welfare_hist.png", dpi=150)
            plt.close()

        if chosen_task_perfs and chosen_total_tokens:
            plt.figure(figsize=(6, 5))
            n = min(len(chosen_task_perfs), len(chosen_total_tokens))
            plt.scatter(chosen_total_tokens[:n], chosen_task_perfs[:n], s=10, alpha=0.6)
            plt.title("Chosen performance vs token usage")
            plt.xlabel("tokens")
            plt.ylabel("task performance")
            plt.tight_layout()
            plt.savefig(figures_dir / "chosen_perf_vs_tokens_scatter.png", dpi=150)
            plt.close()

        logger.info("Figures written: %s", figures_dir)

    except Exception as e:
        logger.warning("Failed to generate figures: %s", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).resolve()

    asyncio.run(run_experiment(cfg, run_dir))


if __name__ == "__main__":
    main()
