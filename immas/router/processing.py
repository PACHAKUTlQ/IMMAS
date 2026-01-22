"""
immas.router.processing

Core request processing logic for the router app.
"""

from __future__ import annotations

import asyncio
import logging
import time

from typing import Any, Callable, Optional

from fastapi import FastAPI

from immas.common.load import AsyncLoadTracker
from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import parse_usage
from immas.router.backend import HttpOpenAIBackend
from immas.router.batching import MicroBatchInfo
from immas.router.logger import AsyncJsonlLogger, RouterBackendScore, RouterLogRecord
from immas.router.predictor import AsyncBackendPredictorPool, PredictorInput
from immas.router.prefix_cache import PrefixMatch, TextPrefixCache, match_prefix
from immas.router.routing import select_backends_round_robin
from immas.router.types import PendingChatCompletion, PreparedChatCompletion
from immas.router.utils import (
    fail_pending_batch,
    should_evict_router_prefix_cache,
    try_set_future_result,
)


_log = logging.getLogger(__name__)


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
                try_set_future_result(
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
            try_set_future_result(
                pending.future,
                (
                    500,
                    {
                        "error": {
                            "message": f"No configured model for backend {backend.backend_id}"
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

        evict_prefix_cache = should_evict_router_prefix_cache(
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

        try_set_future_result(pending.future, (int(status), dict(resp_json)))
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.exception(
            "Unhandled router error processing chat completion (dialogue_id=%s turn=%s)",
            pending.dialogue_id,
            pending.turn_number,
        )
        try_set_future_result(
            pending.future,
            (500, {"error": {"message": "Internal router error"}}),
        )


async def handle_chat_batch(
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

    This function may raise. The batcher-facing wrapper must ensure that in the
    event of an exception, all pending futures in `batch` are resolved.
    """

    if not batch:
        return

    backends: list[HttpOpenAIBackend] = app.state.backends
    backend_model_by_id: dict[str, str] = app.state.backend_model_by_id
    cache: TextPrefixCache = app.state.prefix_cache
    cache_lock: asyncio.Lock = app.state.prefix_cache_lock

    inflight_tasks: set[asyncio.Task[None]] = app.state.inflight_request_tasks

    assigned = await select_backends_round_robin(app, len(batch))
    if len(assigned) != len(batch):
        # Fail all requests if we cannot assign.
        fail_pending_batch(
            batch,
            status_code=503,
            message="No backends available",
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
