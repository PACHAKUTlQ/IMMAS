"""
immas.router.app

A lightweight OpenAI-compatible router that:
- receives /v1/chat/completions
- enqueues requests into a micro-batcher
- micro-batcher forms short batches with strict max-wait bound
- routes each request (currently: round-robin) and forwards to backends
- logs + online-trains predictors based on observed outcomes
- uses backend API keys from its own YAML configuration (no client auth passthrough)

Micro-batching
--------------
The request handler never forwards directly. It enqueues a PendingRequest and
awaits a Future. A background `MicroBatcher`:
- collects up to N requests,
- waits up to T milliseconds after first request,
- then schedules per-request processing tasks.

This provides the "request freezing" foundation for later batch-level auction
routing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from immas.common.load import AsyncLoadTracker
from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import ParsedUsage, parse_usage
from immas.router.backend import HttpOpenAIBackend
from immas.router.batching import MicroBatchInfo, MicroBatcher
from immas.router.config import RouterAppConfig, load_router_app_config
from immas.router.logger import AsyncJsonlLogger, RouterBackendScore, RouterLogRecord
from immas.router.predictor import AsyncBackendPredictorPool, PredictorInput
from immas.router.prefix_cache import PrefixMatch, TextPrefixCache, match_prefix
from immas.router.utils import _get_header, _parse_turn_number


_log = logging.getLogger(__name__)

_HEADER_RUN_ID = "x-immas-run-id"
_HEADER_DIALOGUE_ID = "x-immas-dialogue-id"
_HEADER_TURN_NUMBER = "x-immas-turn-number"
_HEADER_SOURCE = "x-immas-source"

# Conservative eviction heuristic thresholds.
_EVICT_KVMATCH_MIN = 0.8
_EVICT_OBS_CACHE_MAX = 0.10
_EVICT_MIN_PROMPT_TOKENS = 64

ChatCompletionResult = tuple[int, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PendingChatCompletion:
    """One pending /v1/chat/completions request waiting to be processed."""

    run_id: str
    dialogue_id: str
    turn_number: int
    source: str
    body: dict[str, Any]
    t_enqueued_monotonic: float
    future: asyncio.Future[ChatCompletionResult]


@dataclass(frozen=True, slots=True)
class PreparedChatCompletion:
    """
    One queued request with batch metadata and pre-fetched router cache state.

    `cached_text_by_backend` is read under the router cache lock in the batcher,
    so per-request processing does not need to lock for cache *reads*.
    """

    pending: PendingChatCompletion
    assigned_backend: HttpOpenAIBackend
    cached_text_by_backend: dict[str, str | None]
    batch_id: int
    batch_size: int


def _load_cfg_from_env() -> RouterAppConfig:
    """
    Load router config from YAML path in env IMMAS_ROUTER_CONFIG.

    This is the primary configuration mechanism.
    """

    path = (os.environ.get("IMMAS_ROUTER_CONFIG") or "").strip()
    if not path:
        raise RuntimeError(
            "Missing IMMAS_ROUTER_CONFIG. Please set it to a YAML config file path."
        )

    return load_router_app_config(path)


def _should_evict_router_prefix_cache(
    *,
    usage: ParsedUsage,
    turn_number: int,
    kvmatch_text: float,
    obs_cache_ratio: float,
) -> bool:
    """
    Decide whether to evict the router-side prefix cache record for this key.

    We only attempt eviction detection when the backend explicitly reports cached
    token accounting (`usage.cached_tokens_known == True`). Otherwise, cached_tokens=0
    could mean "unreported", and eviction detection would be wrong.

    The heuristic is intentionally conservative: we require near-perfect prefix match,
    sufficiently large prompts, and near-zero observed cache ratio.

    Returns
    -------
    bool
        True if the router should evict its prefix cache record and skip updating it.
    """

    if turn_number <= 1:
        # Eviction only matters when we expected reuse.
        return False

    if not usage.cached_tokens_known:
        return False

    if usage.prompt_tokens < _EVICT_MIN_PROMPT_TOKENS:
        return False

    if float(kvmatch_text) < _EVICT_KVMATCH_MIN:
        return False

    if float(obs_cache_ratio) > _EVICT_OBS_CACHE_MAX:
        return False

    return True


def _try_set_future_result(
    fut: asyncio.Future[ChatCompletionResult], value: ChatCompletionResult
) -> None:
    """
    Best-effort set_result that never raises.

    This protects against races with cancellation / double completion.
    """
    if fut.done():
        return
    try:
        fut.set_result(value)
    except asyncio.InvalidStateError:
        return


async def _select_backends_round_robin(app: FastAPI, n: int) -> list[HttpOpenAIBackend]:
    """
    Select N backends in a single round-robin critical section.

    This is cheaper than acquiring the RR lock per request and also ensures
    deterministic batch ordering.
    """

    backends: list[HttpOpenAIBackend] = app.state.backends
    rr_lock: asyncio.Lock = app.state.rr_lock

    if n <= 0:
        return []
    if not backends:
        return []

    async with rr_lock:
        start_idx: int = int(app.state.rr_index)
        chosen = [backends[(start_idx + i) % len(backends)] for i in range(n)]
        app.state.rr_index = (start_idx + n) % len(backends)

    return chosen


def _task_done_callback_factory(
    inflight: set[asyncio.Task[None]],
) -> Callable[[asyncio.Task[None]], None]:
    """
    Create a done callback that:
    - removes the task from a tracking set,
    - retrieves exceptions to avoid "Task exception was never retrieved".
    """

    def _cb(t: asyncio.Task[None]) -> None:
        inflight.discard(t)
        try:
            _ = t.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            _log.exception("Background request task failed")

    return _cb


async def _process_one_chat_completion(
    prep: PreparedChatCompletion, *, app: FastAPI
) -> None:
    """
    Process exactly one chat completion request and resolve its Future.

    This contains the same core logic as the old inline endpoint:
    - compute per-backend prefix match proxy (using cached router texts)
    - predict per-backend metrics
    - route (currently: pre-assigned round-robin)
    - forward to backend
    - update predictor and router prefix cache on success
    - log RouterLogRecord
    """

    backends: list[HttpOpenAIBackend] = app.state.backends
    backend_model_by_id: dict[str, str] = app.state.backend_model_by_id

    predictors: AsyncBackendPredictorPool = app.state.predictors
    cache: TextPrefixCache = app.state.prefix_cache
    cache_lock: asyncio.Lock = app.state.prefix_cache_lock
    load_tracker: AsyncLoadTracker = app.state.load_tracker
    logger: AsyncJsonlLogger = app.state.logger

    pending = prep.pending

    try:
        messages = pending.body.get("messages")
        prompt_repr = serialize_chat_messages(messages)
        prompt_chars = len(prompt_repr)

        # Compute prefix matches for each backend using cached texts pre-fetched in the batcher.
        pm_by_backend: dict[str, PrefixMatch] = {}
        for b in backends:
            backend_model = backend_model_by_id.get(b.backend_id, "")
            if not backend_model:
                _try_set_future_result(
                    pending.future,
                    (
                        500,
                        {
                            "error": {
                                "message": f"No configured model for backend {b.backend_id}"
                            }
                        },
                    ),
                )
                return

            cached_text = prep.cached_text_by_backend.get(b.backend_id)
            pm_by_backend[b.backend_id] = match_prefix(
                prompt_text=prompt_repr,
                cached_text=cached_text,
            )

        # Chosen backend is pre-assigned
        backend = prep.assigned_backend
        backend_model = backend_model_by_id.get(backend.backend_id, "")
        if not backend_model:
            _try_set_future_result(
                pending.future,
                (
                    500,
                    {
                        "error": {
                            "message": f"No configured model for backend {
                                backend.backend_id
                            }"
                        }
                    },
                ),
            )
            return

        # Use *effective* model for feature computation / caching / logging.
        effective_model = backend_model

        # Predict for all backends under the router load-tracker context.
        async with load_tracker.track() as load:
            inputs_by_backend: dict[str, PredictorInput] = {}
            for b in backends:
                model = backend_model_by_id[b.backend_id]
                pm = pm_by_backend.get(b.backend_id)
                if pm is None:
                    pm = PrefixMatch(
                        ratio=0.0, lcp_chars=0, prompt_chars=0, cached_chars=0
                    )

                inputs_by_backend[b.backend_id] = PredictorInput(
                    backend_id=b.backend_id,
                    model=model,
                    source=pending.source,
                    dialogue_id=pending.dialogue_id,
                    turn_number=pending.turn_number,
                    prompt_repr=prompt_repr,
                    kvmatch_text=float(pm.ratio),
                    router_inflight=int(load.inflight_requests),
                    router_rps_1s=float(load.rps),
                )

            preds_by_backend = await predictors.predict_all(inputs_by_backend)

            # Extract chosen backend prediction fields (for backwards-compatible top-level logging).
            chosen_pred = preds_by_backend.get(backend.backend_id, {})
            pred_latency_ms = float(chosen_pred.get("latency_ms", (0.0, 0.0))[0])
            pred_cost_tokens = float(chosen_pred.get("cost_tokens", (0.0, 0.0))[0])
            pred_perf_prob = float(chosen_pred.get("performance", (0.0, 0.0))[0])
            pred_cache_ratio = float(chosen_pred.get("cache_ratio", (0.0, 0.0))[0])

            chosen_pm = pm_by_backend.get(backend.backend_id) or PrefixMatch(
                ratio=0.0, lcp_chars=0, prompt_chars=0, cached_chars=0
            )
            kvmatch_text = float(chosen_pm.ratio)
            cached_prompt_chars = int(chosen_pm.cached_chars)
            kvmatch_lcp_chars = int(chosen_pm.lcp_chars)

            # Router-controlled headers to backend (no client auth passthrough).
            backend_headers = {
                "X-IMMAS-RUN-ID": pending.run_id,
                "X-IMMAS-DIALOGUE-ID": pending.dialogue_id,
                "X-IMMAS-TURN-NUMBER": str(pending.turn_number),
                "X-IMMAS-SOURCE": pending.source,
            }

            forwarded_body: dict[str, Any] = dict(pending.body)
            forwarded_body["model"] = effective_model

            t0 = time.perf_counter()
            status, resp_json = await backend.forward_chat_completions(
                forwarded_body,
                headers=backend_headers,
            )
            t1 = time.perf_counter()

        obs_latency_ms = (t1 - t0) * 1000.0
        queue_wait_ms = max(0.0, (t0 - pending.t_enqueued_monotonic) * 1000.0)

        completion_id = (
            str(resp_json.get("id") or "") if isinstance(resp_json, dict) else ""
        )

        usage = parse_usage(resp_json if isinstance(resp_json, dict) else {})
        obs_prompt_tokens = usage.prompt_tokens
        obs_completion_tokens = usage.completion_tokens
        obs_total_tokens = usage.total_tokens
        obs_cached_tokens = usage.cached_tokens
        obs_cache_ratio = usage.cache_ratio

        # Placeholder correctness
        correct = True
        error: Optional[str] = None

        evict_prefix_cache = _should_evict_router_prefix_cache(
            usage=usage,
            turn_number=int(pending.turn_number),
            kvmatch_text=float(kvmatch_text),
            obs_cache_ratio=float(obs_cache_ratio),
        )

        # Update predictor + cache only on success (chosen backend only).
        if 200 <= status < 300:
            chosen_inp = inputs_by_backend.get(backend.backend_id)
            if chosen_inp is not None:
                await predictors.update_one(
                    chosen_inp,
                    real_latency_ms=float(obs_latency_ms),
                    real_cost_tokens=int(obs_total_tokens),
                    real_perf_correct=bool(correct),
                )

            if evict_prefix_cache:
                # Backend likely did not have the cached prefix (evicted/disabled);
                # evict router record and skip updating it for this response.
                async with cache_lock:
                    cache.evict(
                        backend_id=backend.backend_id,
                        model=effective_model,
                        dialogue_id=pending.dialogue_id,
                    )
            else:
                # Update router prefix cache using request messages + returned assistant message
                assistant = extract_first_assistant_message(resp_json)
                if assistant is not None and isinstance(messages, list):
                    new_messages = list(messages) + [assistant.to_openai_message()]
                    new_prompt_repr = serialize_chat_messages(new_messages)
                    async with cache_lock:
                        cache.update(
                            backend_id=backend.backend_id,
                            model=effective_model,
                            dialogue_id=pending.dialogue_id,
                            cached_text=new_prompt_repr,
                        )
        else:
            error = f"backend_status={status}"

        # Build per-backend score list for logging (future auction input).
        backend_scores: list[RouterBackendScore] = []
        for b in backends:
            pm = pm_by_backend.get(b.backend_id) or PrefixMatch(
                ratio=0.0, lcp_chars=0, prompt_chars=0, cached_chars=0
            )

            pred = preds_by_backend.get(b.backend_id, {})
            backend_scores.append(
                RouterBackendScore(
                    backend_id=b.backend_id,
                    model=str(backend_model_by_id.get(b.backend_id, "")),
                    cached_prompt_chars=int(pm.cached_chars),
                    kvmatch_lcp_chars=int(pm.lcp_chars),
                    kvmatch_text=float(pm.ratio),
                    pred_latency_ms=float(pred.get("latency_ms", (0.0, 0.0))[0]),
                    pred_cost_tokens=float(pred.get("cost_tokens", (0.0, 0.0))[0]),
                    pred_perf_prob=float(pred.get("performance", (0.0, 0.0))[0]),
                    pred_cache_ratio=float(pred.get("cache_ratio", (0.0, 0.0))[0]),
                )
            )

        # For logging, router inflight/rps from the chosen backend's input (same for all).
        chosen_inp_for_log = inputs_by_backend.get(backend.backend_id)
        router_inflight = (
            int(chosen_inp_for_log.router_inflight) if chosen_inp_for_log else 0
        )
        router_rps_1s = (
            float(chosen_inp_for_log.router_rps_1s) if chosen_inp_for_log else 0.0
        )

        rec = RouterLogRecord(
            run_id=pending.run_id,
            t_start_monotonic=float(t0),
            t_end_monotonic=float(t1),
            batch_id=int(prep.batch_id),
            batch_size=int(prep.batch_size),
            queue_wait_ms=float(queue_wait_ms),
            backend_id=backend.backend_id,
            backend_base_url_v1=backend.base_url_v1,
            model=effective_model,
            source=pending.source,
            dialogue_id=pending.dialogue_id,
            turn_number=int(pending.turn_number),
            prompt_chars=int(prompt_chars),
            cached_prompt_chars=int(cached_prompt_chars),
            kvmatch_lcp_chars=int(kvmatch_lcp_chars),
            kvmatch_text=float(kvmatch_text),
            router_inflight=int(router_inflight),
            router_rps_1s=float(router_rps_1s),
            pred_latency_ms=float(pred_latency_ms),
            pred_cost_tokens=float(pred_cost_tokens),
            pred_perf_prob=float(pred_perf_prob),
            pred_cache_ratio=float(pred_cache_ratio),
            backend_scores=backend_scores,
            completion_id=completion_id,
            obs_latency_ms=float(obs_latency_ms),
            obs_prompt_tokens=int(obs_prompt_tokens),
            obs_completion_tokens=int(obs_completion_tokens),
            obs_total_tokens=int(obs_total_tokens),
            obs_cached_tokens=int(obs_cached_tokens),
            obs_cache_ratio=float(obs_cache_ratio),
            correct=bool(correct),
            error=error,
        )
        await logger.log(rec)

        _try_set_future_result(pending.future, (int(status), dict(resp_json)))
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.exception(
            "Unhandled router error processing chat completion (dialogue_id=%s turn=%s)",
            pending.dialogue_id,
            pending.turn_number,
        )
        _try_set_future_result(
            pending.future,
            (500, {"error": {"message": "Internal router error"}}),
        )


async def _handle_chat_batch(
    app: FastAPI,
    batch: list[PendingChatCompletion],
    info: MicroBatchInfo,
) -> None:
    """
    Batch handler invoked by MicroBatcher.

    Responsibilities:
    - choose an assigned backend for each request (current: round-robin)
    - fetch router prefix cache texts for all requests/backends under one lock
    - schedule per-request processing tasks
    """

    if not batch:
        return

    backends: list[HttpOpenAIBackend] = app.state.backends
    backend_model_by_id: dict[str, str] = app.state.backend_model_by_id
    cache: TextPrefixCache = app.state.prefix_cache
    cache_lock: asyncio.Lock = app.state.prefix_cache_lock

    inflight_tasks: set[asyncio.Task[None]] = app.state.inflight_request_tasks

    assigned = await _select_backends_round_robin(app, len(batch))
    if len(assigned) != len(batch):
        # Fail all requests if we cannot assign.
        for p in batch:
            _try_set_future_result(
                p.future,
                (503, {"error": {"message": "No backends available"}}),
            )
        return

    # Prefetch cached texts for all (request, backend) pairs under one lock.
    cached_by_req: list[dict[str, str | None]] = [{} for _ in batch]
    async with cache_lock:
        for i, pending in enumerate(batch):
            for b in backends:
                model = backend_model_by_id.get(b.backend_id, "")
                if not model:
                    cached_by_req[i][b.backend_id] = None
                    continue
                cached_by_req[i][b.backend_id] = cache.get(
                    backend_id=b.backend_id,
                    model=model,
                    dialogue_id=pending.dialogue_id,
                )

    done_cb = _task_done_callback_factory(inflight_tasks)

    for i, pending in enumerate(batch):
        prep = PreparedChatCompletion(
            pending=pending,
            assigned_backend=assigned[i],
            cached_text_by_backend=cached_by_req[i],
            batch_id=int(info.batch_id),
            batch_size=int(info.batch_size),
        )
        t = asyncio.create_task(
            _process_one_chat_completion(prep, app=app),
            name=f"chat_completion.batch{info.batch_id}.i{i}",
        )
        inflight_tasks.add(t)
        t.add_done_callback(done_cb)


def create_app() -> FastAPI:
    cfg = _load_cfg_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Router feature cache (text prefix) - protect with lock.
        app.state.prefix_cache = TextPrefixCache()
        app.state.prefix_cache_lock = asyncio.Lock()

        # Router-side load tracker (inflight + RPS).
        app.state.load_tracker = AsyncLoadTracker(window_s=1.0)

        # Backends registry (ordered list for round-robin).
        app.state.backends = [
            HttpOpenAIBackend(
                backend_id=b.backend_id,
                base_url_v1=b.base_url_v1,
                api_key=b.api_key,
            )
            for b in cfg.backends
        ]
        app.state.backend_model_by_id = {b.backend_id: b.model for b in cfg.backends}

        # Independent predictor per backend.
        app.state.predictors = AsyncBackendPredictorPool(
            backend_ids=[b.backend_id for b in cfg.backends]
        )

        # Round-robin state.
        app.state.rr_lock = asyncio.Lock()
        app.state.rr_index = 0

        # Routing policy string (future: auction, etc.)
        app.state.routing_policy = cfg.router.routing

        # JSONL logger
        app.state.logger = AsyncJsonlLogger(
            cfg.router.log_path, append=cfg.router.log_append, flush_every=1
        )
        await app.state.logger.__aenter__()

        # Track inflight request tasks (so shutdown can cancel/await them).
        app.state.inflight_request_tasks = set()

        # Micro-batcher
        batching_cfg = cfg.router.batching
        if not batching_cfg.enabled:
            raise RuntimeError(
                "router.batching.enabled is false, but this router version requires batching. "
                "Set router.batching.enabled: true (or remove the key to use defaults)."
            )

        async def _batch_handler(
            batch: list[PendingChatCompletion], info: MicroBatchInfo
        ) -> None:
            await _handle_chat_batch(app, batch, info)

        app.state.chat_batcher = MicroBatcher[PendingChatCompletion](
            max_batch_size=int(batching_cfg.max_batch_size),
            max_wait_ms=float(batching_cfg.max_wait_ms),
            max_queue_size=int(batching_cfg.max_queue_size),
            handler=_batch_handler,
            name="immas.chat_completions.microbatcher",
        )
        await app.state.chat_batcher.start()

        yield

        # Shutdown order matters:
        # 1) stop accepting new work (batcher)
        # 2) cancel/await inflight request tasks
        # 3) close logger
        # 4) close backends

        batcher: MicroBatcher[PendingChatCompletion] = app.state.chat_batcher
        await batcher.close()

        inflight: set[asyncio.Task[None]] = app.state.inflight_request_tasks
        if inflight:
            for t in list(inflight):
                t.cancel()
            await asyncio.gather(*list(inflight), return_exceptions=True)
            inflight.clear()

        await app.state.logger.close()
        for b in app.state.backends:
            await b.close()

    app = FastAPI(title="IMMAS Router", version="0.5", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models(req: Request) -> JSONResponse:
        """
        Return the union of router-configured backend model names.
        """

        backend_model_by_id: dict[str, str] = req.app.state.backend_model_by_id
        seen: set[str] = set()
        data: list[dict[str, Any]] = []
        for m in backend_model_by_id.values():
            if m in seen:
                continue
            seen.add(m)
            data.append({"id": m, "object": "model"})

        return JSONResponse(status_code=200, content={"object": "list", "data": data})

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: Request) -> JSONResponse:
        """
        Enqueue the request into the micro-batcher and await completion.
        """

        run_id = _get_header(req, _HEADER_RUN_ID) or "run_unknown"
        dialogue_id = _get_header(req, _HEADER_DIALOGUE_ID) or "dialogue_unknown"
        turn_number = _parse_turn_number(_get_header(req, _HEADER_TURN_NUMBER))
        source = _get_header(req, _HEADER_SOURCE) or "unknown"

        try:
            body: Any = await req.json()
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": {"message": "Invalid JSON body"}}
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"error": {"message": "Invalid JSON body"}}
            )

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ChatCompletionResult] = loop.create_future()

        pending = PendingChatCompletion(
            run_id=run_id,
            dialogue_id=dialogue_id,
            turn_number=int(turn_number),
            source=source,
            body=dict(body),
            t_enqueued_monotonic=float(time.perf_counter()),
            future=fut,
        )

        batcher: MicroBatcher[PendingChatCompletion] = req.app.state.chat_batcher
        ok = batcher.try_submit(pending)
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Router overloaded (batch queue full)"}},
            )

        status, resp_json = await fut
        return JSONResponse(status_code=int(status), content=resp_json)

    return app


app = create_app()
