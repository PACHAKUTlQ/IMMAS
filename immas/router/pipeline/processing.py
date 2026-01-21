"""
immas.router.pipeline.processing

Core request processing logic for the router app.
"""

from __future__ import annotations

import asyncio
import logging
import time

from typing import Any, Callable, Optional

from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import parse_usage
from immas.router.components.batching import MicroBatchInfo
from immas.router.components.logger import RouterBackendScore, RouterLogRecord
from immas.router.components.predictor import PredictorInput
from immas.router.components.prefix_cache import PrefixMatch, match_prefix
from immas.router.pipeline.routing import select_backends_round_robin
from immas.router.state import RouterState
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


def _find_chosen_backend_score(
    backend_scores: list[RouterBackendScore], *, backend_id: str
) -> RouterBackendScore | None:
    for s in backend_scores:
        if s.backend_id == backend_id:
            return s
    return None


async def _process_one_chat_completion(
    prep: PreparedChatCompletion, *, state: RouterState
) -> None:
    """
    Process exactly one chat completion request and resolve its Future.

    Batch-level scoring is already done in `handle_chat_batch()` and passed in via
    PreparedChatCompletion. This function focuses on:
    - forwarding to backend
    - measuring observations
    - updating predictor and router prefix cache on success
    - logging
    """

    pending = prep.pending
    backend = prep.assigned_backend
    effective_model = prep.effective_model

    chosen_score = _find_chosen_backend_score(
        prep.backend_scores, backend_id=backend.backend_id
    )
    if chosen_score is None:
        try_set_future_result(
            pending.future,
            (
                500,
                {"error": {"message": "Internal router error: missing chosen score"}},
            ),
        )
        return

    kvmatch_text = float(chosen_score.kvmatch_text)
    cached_prompt_chars = int(chosen_score.cached_prompt_chars)
    kvmatch_lcp_chars = int(chosen_score.kvmatch_lcp_chars)

    pred_latency_ms = float(chosen_score.pred_latency_ms)
    pred_cost_tokens = float(chosen_score.pred_cost_tokens)
    pred_perf_prob = float(chosen_score.pred_perf_prob)
    pred_cache_ratio = float(chosen_score.pred_cache_ratio)

    try:
        messages = pending.body.get("messages")
        prompt_chars = int(prep.prompt_chars)

        backend_headers = {
            "X-IMMAS-RUN-ID": pending.run_id,
            "X-IMMAS-DIALOGUE-ID": pending.dialogue_id,
            "X-IMMAS-TURN-NUMBER": str(pending.turn_number),
            "X-IMMAS-SOURCE": pending.source,
        }

        forwarded_body: dict[str, Any] = dict(pending.body)
        forwarded_body["model"] = effective_model

        async with state.load_tracker.track() as load:
            t0 = time.perf_counter()
            status, resp_json = await backend.forward_chat_completions(
                forwarded_body,
                headers=backend_headers,
            )
            t1 = time.perf_counter()

            # Note: load fields are sampled during the tracked section.
            # The logged RouterLogRecord fields currently represent decision-time
            # features, which are computed in batch handler. We therefore do not
            # overwrite those fields here.
            _ = load

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

        correct = True
        error: Optional[str] = None

        evict_prefix_cache = should_evict_router_prefix_cache(
            usage=usage,
            turn_number=int(pending.turn_number),
            kvmatch_text=float(kvmatch_text),
            obs_cache_ratio=float(obs_cache_ratio),
        )

        if 200 <= status < 300:
            # Update predictor for the chosen backend only.
            await state.predictors.update_one(
                prep.chosen_predictor_input,
                real_latency_ms=float(obs_latency_ms),
                real_cost_tokens=int(obs_total_tokens),
                real_perf_correct=bool(correct),
            )

            # Update/evict router prefix cache for the chosen backend only.
            if evict_prefix_cache:
                async with state.prefix_cache_lock:
                    state.prefix_cache.evict(
                        backend_id=backend.backend_id,
                        model=effective_model,
                        dialogue_id=pending.dialogue_id,
                    )
            else:
                assistant = extract_first_assistant_message(resp_json)
                if assistant is not None and isinstance(messages, list):
                    new_messages = list(messages) + [assistant.to_openai_message()]
                    new_prompt_repr = serialize_chat_messages(new_messages)
                    async with state.prefix_cache_lock:
                        state.prefix_cache.update(
                            backend_id=backend.backend_id,
                            model=effective_model,
                            dialogue_id=pending.dialogue_id,
                            cached_text=new_prompt_repr,
                        )
        else:
            error = f"backend_status={status}"

        # Router load fields in log record should reflect routing-time features.
        # Currently they are computed in the batch handler and embedded into the
        # chosen_predictor_input.
        router_inflight = int(prep.chosen_predictor_input.router_inflight)
        router_rps_1s = float(prep.chosen_predictor_input.router_rps_1s)

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
            backend_scores=list(prep.backend_scores),
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
        await state.logger.log(rec)

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
    state: RouterState,
    batch: list[PendingChatCompletion],
    info: MicroBatchInfo,
) -> None:
    """
    Batch handler invoked by MicroBatcher.

    Responsibilities (batch-level):
    - choose an assigned backend for each request (current: round-robin)
    - fetch router prefix cache texts for all requests/backends under one lock
    - compute per-backend prefix match + predictor outputs for each request
    - schedule per-request processing tasks

    Notes
    -----
    This function must remain reasonably fast because the MicroBatcher awaits it.
    """

    if not batch:
        return

    assigned = await select_backends_round_robin(state, len(batch))
    if len(assigned) != len(batch):
        fail_pending_batch(
            batch,
            status_code=503,
            message="No backends available",
        )
        return

    # Serialize prompts (fail individual bad requests early).
    prompt_repr_by_i: list[str | None] = [None] * len(batch)
    prompt_chars_by_i: list[int] = [0] * len(batch)

    for i, pending in enumerate(batch):
        try:
            messages = pending.body.get("messages")
            pr = serialize_chat_messages(messages)
        except Exception:
            try_set_future_result(
                pending.future,
                (400, {"error": {"message": "Invalid chat messages"}}),
            )
            continue
        prompt_repr_by_i[i] = pr
        prompt_chars_by_i[i] = int(len(pr))

    # Read cached texts under one lock: cached_text[i][backend_id] -> str|None
    cached_text_by_req: list[dict[str, str | None]] = [{} for _ in batch]
    async with state.prefix_cache_lock:
        for i, pending in enumerate(batch):
            for b in state.backends:
                model = state.backend_model_by_id.get(b.backend_id, "")
                if not model:
                    cached_text_by_req[i][b.backend_id] = None
                    continue
                cached_text_by_req[i][b.backend_id] = state.prefix_cache.get(
                    backend_id=b.backend_id,
                    model=model,
                    dialogue_id=pending.dialogue_id,
                )

    # For each request, compute prefix match + predictor inputs and run predictors.
    # We keep decision-time router load features as currently available (no extra
    # load tracking in batch handler). These are set to zeros for now.
    router_inflight_decision = 0
    router_rps_1s_decision = 0.0

    pm_by_req: list[dict[str, PrefixMatch]] = [{} for _ in batch]
    inputs_by_req: list[dict[str, PredictorInput]] = [{} for _ in batch]

    for i, pending in enumerate(batch):
        pr = prompt_repr_by_i[i]
        if pr is None:
            # Already failed above.
            continue

        for b in state.backends:
            model = state.backend_model_by_id.get(b.backend_id, "")
            cached_text = cached_text_by_req[i].get(b.backend_id)
            pm = match_prefix(prompt_text=pr, cached_text=cached_text)
            pm_by_req[i][b.backend_id] = pm

            # Predictor input for this (request, backend)
            inputs_by_req[i][b.backend_id] = PredictorInput(
                backend_id=b.backend_id,
                model=model,
                source=pending.source,
                dialogue_id=pending.dialogue_id,
                turn_number=int(pending.turn_number),
                prompt_repr=pr,
                kvmatch_text=float(pm.ratio),
                router_inflight=int(router_inflight_decision),
                router_rps_1s=float(router_rps_1s_decision),
            )

    async def _predict_for_i(i: int) -> dict[str, dict[str, tuple[float, float]]]:
        # If prompt serialization failed, there are no inputs to score.
        if not inputs_by_req[i]:
            return {}
        return await state.predictors.predict_all(inputs_by_req[i])

    preds_by_req = await asyncio.gather(
        *[_predict_for_i(i) for i in range(len(batch))],
        return_exceptions=False,
    )

    # Schedule per-request tasks with PreparedChatCompletion
    done_cb = _task_done_callback_factory(state.inflight_request_tasks)

    for i, pending in enumerate(batch):
        if pending.future.done():
            continue  # already failed (e.g. invalid messages)

        backend = assigned[i]
        backend_model = state.backend_model_by_id.get(backend.backend_id, "")
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
            continue

        pr = prompt_repr_by_i[i]
        if pr is None:
            # Should already be done.
            try_set_future_result(
                pending.future,
                (400, {"error": {"message": "Invalid chat messages"}}),
            )
            continue

        # Build RouterBackendScore list (stable order of state.backends).
        backend_scores: list[RouterBackendScore] = []
        pred_map = preds_by_req[i] if isinstance(preds_by_req[i], dict) else {}
        pm_map = pm_by_req[i]

        for b in state.backends:
            pm = pm_map.get(b.backend_id) or PrefixMatch(
                ratio=0.0, lcp_chars=0, prompt_chars=0, cached_chars=0
            )
            pred = pred_map.get(b.backend_id, {})
            backend_scores.append(
                RouterBackendScore(
                    backend_id=b.backend_id,
                    model=str(state.backend_model_by_id.get(b.backend_id, "")),
                    cached_prompt_chars=int(pm.cached_chars),
                    kvmatch_lcp_chars=int(pm.lcp_chars),
                    kvmatch_text=float(pm.ratio),
                    pred_latency_ms=float(pred.get("latency_ms", (0.0, 0.0))[0]),
                    pred_cost_tokens=float(pred.get("cost_tokens", (0.0, 0.0))[0]),
                    pred_perf_prob=float(pred.get("performance", (0.0, 0.0))[0]),
                    pred_cache_ratio=float(pred.get("cache_ratio", (0.0, 0.0))[0]),
                )
            )

        chosen_inp = inputs_by_req[i].get(backend.backend_id)
        if chosen_inp is None:
            try_set_future_result(
                pending.future,
                (
                    500,
                    {
                        "error": {
                            "message": "Internal router error: missing predictor input"
                        }
                    },
                ),
            )
            continue

        prep = PreparedChatCompletion(
            pending=pending,
            assigned_backend=backend,
            effective_model=str(backend_model),
            prompt_repr=str(pr),
            prompt_chars=int(prompt_chars_by_i[i]),
            backend_scores=backend_scores,
            chosen_predictor_input=chosen_inp,
            batch_id=int(info.batch_id),
            batch_size=int(info.batch_size),
        )
        t = asyncio.create_task(
            _process_one_chat_completion(prep, state=state),
            name=f"chat_completion.batch{info.batch_id}.i{i}",
        )
        state.inflight_request_tasks.add(t)
        t.add_done_callback(done_cb)
