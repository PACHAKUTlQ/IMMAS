"""
immas.router.pipeline.processing

Core request processing logic for the router app.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from typing import Any, Callable, Optional

from immas.common.load import LoadSnapshot
from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import parse_usage
from immas.router.auction.mechanism import AuctionParams, compute_welfare
from immas.router.components.batching import MicroBatchInfo
from immas.router.components.detailed_csv import RouterDetailedCsvRow
from immas.router.components.logger import RouterBackendScore, RouterLogRecord
from immas.router.components.performance import (
    PerformanceEvalContext,
    RougeCoqaEvaluator,
    RougeScoreResult,
    TokenSpanCoqaEvaluator,
    TokenSpanScoreResult,
)
from immas.router.components.performance.normalization import extract_last_nonempty_line
from immas.router.components.predictor import PredictorInput
from immas.router.components.prefix_cache import PrefixMatch, match_prefix
from immas.router.pipeline.routing import (
    select_backends_auction,
    select_backends_round_robin,
)
from immas.router.pricing import BackendTokenPrices, compute_observed_cost_tokens
from immas.router.state import RouterState
from immas.router.telemetry import pop_ttft_monotonic
from immas.router.types import PendingChatCompletion, PreparedChatCompletion
from immas.router.utils import (
    fail_pending_batch,
    should_evict_router_prefix_cache,
    try_set_future_result,
)


_log = logging.getLogger(__name__)

_Q_TURN_PREFIX_RE = re.compile(r"^\s*Q\s*\d+\s*:\s*", re.IGNORECASE)


def _extract_story_and_question(messages: Any) -> tuple[str, str]:
    """
    Best-effort extraction of (story, question) from OpenAI chat messages.

    Expected CoQA-shaped prompt style:
    - system: ...
    - user: "Story (...):\\n<story>"
    - user: "Q{turn}: <question>"
    - assistant: ...
    - user: "Q{turn+1}: <question>"

    Returns empty strings if extraction fails.
    """
    if not isinstance(messages, list):
        return "", ""

    story = ""
    question = ""

    # Story: first user message that looks like a story container.
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        s = c.strip()
        if not s:
            continue
        if s.lower().startswith("story"):
            # If "Story ...:\n<story>", take the part after the first newline if present.
            if "\n" in s:
                story = s.split("\n", 1)[1].strip()
            else:
                story = s
            break

    # Question: last user message content.
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        q = c.strip()
        if not q:
            continue
        q = _Q_TURN_PREFIX_RE.sub("", q).strip()
        question = q
        break

    return story, question


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


async def _maybe_log_detailed_csv(
    *,
    state: RouterState,
    pending: PendingChatCompletion,
    forwarded_body: dict[str, Any],
    resp_json: dict[str, Any] | None,
    perf_details: TokenSpanScoreResult | RougeScoreResult | None,
    correct: bool,
) -> None:
    """
    Best-effort detailed CSV logging.

    Produces a clean row with context, answers, and performance evaluation details.
    The logged fields depend on the configured performance evaluator.

    Never raises.
    """

    logger = state.detailed_csv_logger
    if not logger:
        return

    try:
        messages = forwarded_body.get("messages")
        story, question = _extract_story_and_question(messages)

        llm_answer = ""
        llm_answer_last_line = ""
        if isinstance(resp_json, dict):
            assistant = extract_first_assistant_message(resp_json)
            if assistant and assistant.content:
                llm_answer = str(assistant.content)
                llm_answer_last_line = extract_last_nonempty_line(llm_answer)

        evaluator_name = state.cfg.router.performance.evaluator
        gold_answer = ""
        token_span_matched: bool | None = None
        rouge_metric_used: str | None = None
        rouge_used_f1: float | None = None
        rouge_1_f1: float | None = None
        rouge_2_f1: float | None = None
        rouge_l_f1: float | None = None

        if isinstance(perf_details, TokenSpanScoreResult):
            gold_answer = perf_details.reference_raw
            token_span_matched = perf_details.matched
        elif isinstance(perf_details, RougeScoreResult):
            gold_answer = perf_details.reference_raw
            rouge_metric_used = str(perf_details.metric_used)
            rouge_used_f1 = float(perf_details.used_f1)
            rouge_1_f1 = float(perf_details.f1.rouge_1_f1)
            rouge_2_f1 = float(perf_details.f1.rouge_2_f1)
            rouge_l_f1 = float(perf_details.f1.rouge_l_f1)

        row = RouterDetailedCsvRow(
            source=str(pending.source),
            dialogue_id=str(pending.dialogue_id),
            turn_number=int(pending.turn_number),
            story=story,
            question=question,
            gold_answer=gold_answer,
            llm_answer_last_line=llm_answer_last_line,
            llm_answer=llm_answer,
            evaluator=evaluator_name,
            correct=correct,
            token_span_matched=token_span_matched,
            rouge_metric_used=rouge_metric_used,
            rouge_used_f1=rouge_used_f1,
            rouge_1_f1=rouge_1_f1,
            rouge_2_f1=rouge_2_f1,
            rouge_l_f1=rouge_l_f1,
        )
        await logger.log(row)
    except Exception:
        _log.exception("Failed to enqueue detailed CSV row")


async def _process_one_chat_completion(
    prep: PreparedChatCompletion, *, state: RouterState
) -> None:
    """
    Process exactly one chat completion request and resolve its Future.

    Batch-level scoring and routing is done in `handle_chat_batch()`. This function:
    - enforces per-backend concurrency via semaphore
    - forwards to backend
    - measures observations
    - updates predictor and router prefix cache on success
    - logs

    Notes
    -----
    This router treats "latency" as a TTFT-like proxy when upstream streaming is
    enabled. Specifically:
    - obs_latency_ms is computed from (t_first_stream_chunk - t_start_monotonic),
    - t_end_monotonic in the JSONL log record is also set to that TTFT timestamp,
      so downstream code that computes latency from timestamps sees TTFT too.

    The client-visible request latency remains end-to-end completion time because
    the router must consume the full stream to return a non-streaming JSON response.
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

        sem = state.backend_semaphores.get(backend.backend_id) or asyncio.Semaphore(1)
        backend_tracker = state.backend_load_trackers.get(backend.backend_id)

        t0 = time.monotonic()
        async with sem:
            # Track both router-global and backend-local inflight.
            if backend_tracker is None:
                async with state.load_tracker.track():
                    status, resp_json = await backend.forward_chat_completions(
                        forwarded_body,
                        headers=backend_headers,
                    )
            else:
                async with state.load_tracker.track(), backend_tracker.track():
                    status, resp_json = await backend.forward_chat_completions(
                        forwarded_body,
                        headers=backend_headers,
                    )
        t_complete = time.monotonic()

        resp_json_dict = resp_json if isinstance(resp_json, dict) else None

        # Strip internal telemetry from payload before any external exposure.
        t_first_token_mon = pop_ttft_monotonic(resp_json_dict)

        # Latency is TTFT-like when available, else completion time.
        if t_first_token_mon is not None:
            obs_latency_ms = max(0.0, (t_first_token_mon - t0) * 1000.0)
            t_end_for_log = float(t_first_token_mon)
        else:
            obs_latency_ms = (t_complete - t0) * 1000.0
            t_end_for_log = float(t_complete)

        queue_wait_ms = max(0.0, (t0 - pending.t_enqueued_monotonic) * 1000.0)

        completion_id = (
            str(resp_json_dict.get("id") or "")
            if isinstance(resp_json_dict, dict)
            else ""
        )

        usage = parse_usage(resp_json_dict if isinstance(resp_json_dict, dict) else {})
        obs_prompt_tokens = usage.prompt_tokens
        obs_completion_tokens = usage.completion_tokens
        obs_total_tokens = usage.total_tokens
        obs_cached_tokens = usage.cached_tokens
        obs_cache_ratio = usage.cache_ratio

        prices = (
            state.backend_prices_by_id.get(backend.backend_id) or BackendTokenPrices()
        )
        obs_cost_tokens = compute_observed_cost_tokens(usage=usage, prices=prices)

        error: Optional[str] = None
        correct = False
        perf_details: TokenSpanScoreResult | RougeScoreResult | None = None

        if 200 <= int(status) < 300 and resp_json_dict is not None:
            ctx = PerformanceEvalContext(
                run_id=pending.run_id,
                dialogue_id=pending.dialogue_id,
                turn_number=int(pending.turn_number),
                source=pending.source,
                request_body=forwarded_body,
                response_json=resp_json_dict,
            )

            # Get detailed score if evaluator supports it, then get correctness.
            if isinstance(state.perf_evaluator, TokenSpanCoqaEvaluator):
                perf_details = state.perf_evaluator.score(ctx)
                correct = bool(perf_details.matched) if perf_details else False
            elif isinstance(state.perf_evaluator, RougeCoqaEvaluator):
                perf_details = state.perf_evaluator.score(ctx)
                correct = bool(perf_details.correct) if perf_details else False
            else:
                correct = bool(state.perf_evaluator.evaluate(ctx))

        else:
            error = f"backend_status={status}"

        # Optional clean CSV row (best-effort).
        await _maybe_log_detailed_csv(
            state=state,
            pending=pending,
            forwarded_body=forwarded_body,
            resp_json=resp_json_dict,
            perf_details=perf_details,
            correct=correct,
        )

        evict_prefix_cache = should_evict_router_prefix_cache(
            usage=usage,
            turn_number=int(pending.turn_number),
            kvmatch_text=float(chosen_score.kvmatch_text),
            obs_cache_ratio=float(obs_cache_ratio),
        )

        if 200 <= int(status) < 300:
            await state.predictors.update_one(
                prep.chosen_predictor_input,
                real_latency_ms=float(obs_latency_ms),
                real_cost_tokens=float(obs_cost_tokens),
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
                assistant = extract_first_assistant_message(resp_json_dict or {})
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

        # Router load fields in log record should reflect routing-time features.
        # Currently they are computed in the batch handler and embedded into the
        # chosen_predictor_input.
        router_inflight = int(prep.chosen_predictor_input.router_inflight)
        router_rps_1s = float(prep.chosen_predictor_input.router_rps_1s)

        rec = RouterLogRecord(
            run_id=pending.run_id,
            t_start_monotonic=float(t0),
            t_end_monotonic=float(t_end_for_log),
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
            cached_prompt_chars=int(chosen_score.cached_prompt_chars),
            kvmatch_lcp_chars=int(chosen_score.kvmatch_lcp_chars),
            kvmatch_text=float(chosen_score.kvmatch_text),
            router_inflight=int(router_inflight),
            router_rps_1s=float(router_rps_1s),
            pred_latency_ms=float(chosen_score.pred_latency_ms),
            pred_cost_tokens=float(chosen_score.pred_cost_tokens),
            pred_perf_prob=float(chosen_score.pred_perf_prob),
            pred_cache_ratio=float(chosen_score.pred_cache_ratio),
            routing_policy=str(prep.routing_policy),
            auction_matched=bool(prep.auction_matched),
            auction_total_welfare=float(prep.auction_total_welfare),
            chosen_client_valuation=float(prep.chosen_client_valuation),
            chosen_base_cost=float(prep.chosen_base_cost),
            chosen_welfare=float(prep.chosen_welfare),
            vcg_fee=prep.vcg_fee,
            vcg_total_payment=prep.vcg_total_payment,
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

        # Best-effort JSONL logging (should not fail requests).
        try:
            await state.logger.log(rec)
        except Exception:
            _log.exception("Failed to write JSONL router log record")

        try_set_future_result(pending.future, (int(status), dict(resp_json_dict or {})))
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
    - serialize prompts
    - read router prefix cache texts for all requests/backends under one lock
    - compute per-backend prefix match + predictor outputs for each request
    - run routing policy (round robin or auction) using the whole batch
    - schedule per-request processing tasks
    """

    if not batch:
        return
    if not state.backends:
        fail_pending_batch(batch, status_code=503, message="No backends available")
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
                cached_text_by_req[i][b.backend_id] = (
                    state.prefix_cache.get(
                        backend_id=b.backend_id,
                        model=model,
                        dialogue_id=pending.dialogue_id,
                    )
                    if model
                    else None
                )

    # Decision-time load snapshots.
    load_snap = await state.load_tracker.snapshot()
    router_inflight_decision = int(load_snap.inflight_requests)
    router_rps_1s_decision = float(load_snap.rps)

    async def _backend_snap(backend_id: str) -> LoadSnapshot:
        tr = state.backend_load_trackers.get(backend_id)
        if tr is None:
            # Defensive fallback
            return LoadSnapshot(
                inflight_requests=0,
                rps=0.0,
                window_s=1.0,
                t_monotonic=float(time.monotonic()),
            )
        return await tr.snapshot()

    backend_ids = [b.backend_id for b in state.backends]
    backend_snaps = await asyncio.gather(
        *[_backend_snap(bid) for bid in backend_ids],
        return_exceptions=False,
    )
    backend_inflight_by_id = {
        bid: int(s.inflight_requests) for bid, s in zip(backend_ids, backend_snaps)
    }
    backend_rps_by_id = {
        bid: float(s.rps) for bid, s in zip(backend_ids, backend_snaps)
    }
    backend_cap_by_id = {
        bid: max(1, int(state.backend_capacity_by_id.get(bid, 1)))
        for bid in backend_ids
    }

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

            bid = b.backend_id

            # Predictor input for this (request, backend)
            inputs_by_req[i][bid] = PredictorInput(
                backend_id=bid,
                model=model,
                source=pending.source,
                dialogue_id=pending.dialogue_id,
                turn_number=int(pending.turn_number),
                prompt_repr=pr,
                kvmatch_text=float(pm.ratio),
                router_inflight=int(router_inflight_decision),
                router_rps_1s=float(router_rps_1s_decision),
                backend_inflight=int(backend_inflight_by_id.get(bid, 0)),
                backend_rps_1s=float(backend_rps_by_id.get(bid, 0.0)),
                backend_capacity=int(backend_cap_by_id.get(bid, 1)),
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

    # Auction params for computing per-backend welfare fields in RouterBackendScore.
    auc_cfg = state.cfg.router.auction
    auc_params = AuctionParams(
        quality_scale=float(auc_cfg.quality_scale),
        latency_scale=float(auc_cfg.latency_scale),
        cost_scale=float(auc_cfg.cost_scale),
        delta_default=float(auc_cfg.delta_default),
        min_welfare_edge=float(auc_cfg.min_welfare_edge),
        mcmf_scale=int(auc_cfg.mcmf_scale),
    )
    delta = float(auc_params.delta_default)

    backend_scores_by_req: list[list[RouterBackendScore]] = [[] for _ in batch]
    for i, pending in enumerate(batch):
        if prompt_repr_by_i[i] is None:
            continue
        pred_map = preds_by_req[i] if isinstance(preds_by_req[i], dict) else {}
        pm_map = pm_by_req[i]

        scores: list[RouterBackendScore] = []
        for b in state.backends:
            pm = pm_map.get(b.backend_id) or PrefixMatch(
                ratio=0.0, lcp_chars=0, prompt_chars=0, cached_chars=0
            )
            pred = pred_map.get(b.backend_id, {})
            pred_latency_ms = float(pred.get("latency_ms", (0.0, 0.0))[0])
            pred_cost_tokens = float(pred.get("cost_tokens", (0.0, 0.0))[0])
            pred_perf_prob = float(pred.get("performance", (0.0, 0.0))[0])
            pred_cache_ratio = float(pred.get("cache_ratio", (0.0, 0.0))[0])

            welfare, client_val, base_cost = compute_welfare(
                delta=float(delta),
                pred_latency_ms=float(pred_latency_ms),
                pred_cost_tokens=float(pred_cost_tokens),
                pred_perf_prob=float(pred_perf_prob),
                params=auc_params,
            )

            scores.append(
                RouterBackendScore(
                    backend_id=b.backend_id,
                    model=str(state.backend_model_by_id.get(b.backend_id, "")),
                    cached_prompt_chars=int(pm.cached_chars),
                    kvmatch_lcp_chars=int(pm.lcp_chars),
                    kvmatch_text=float(pm.ratio),
                    pred_latency_ms=float(pred_latency_ms),
                    pred_cost_tokens=float(pred_cost_tokens),
                    pred_perf_prob=float(pred_perf_prob),
                    pred_cache_ratio=float(pred_cache_ratio),
                    client_valuation=float(client_val),
                    base_cost=float(base_cost),
                    welfare=float(welfare),
                )
            )
        backend_scores_by_req[i] = scores

    # Active indices = requests not already failed early.
    active_indices: list[int] = [
        i
        for i, p in enumerate(batch)
        if (not p.future.done()) and (prompt_repr_by_i[i] is not None)
    ]

    # Choose backends for active requests.
    assigned_by_i: list[Optional[Any]] = [None] * len(batch)
    auction_matched_by_i: list[bool] = [False] * len(batch)
    auction_total_welfare = 0.0
    vcg_fee_by_i: list[Optional[float]] = [None] * len(batch)
    vcg_pay_by_i: list[Optional[float]] = [None] * len(batch)

    if state.routing_policy == "auction":
        dec = await select_backends_auction(
            state,
            active_indices=active_indices,
            backend_scores_by_req=backend_scores_by_req,
        )
        assigned_by_i = dec.assigned_by_i
        auction_matched_by_i = dec.auction_matched_by_i
        auction_total_welfare = float(dec.auction_total_welfare)
        vcg_fee_by_i = dec.vcg_fee_by_i
        vcg_pay_by_i = dec.vcg_total_payment_by_i
    else:
        chosen = await select_backends_round_robin(state, len(active_indices))
        for idx, backend in zip(active_indices, chosen):
            assigned_by_i[idx] = backend

    # Schedule per-request tasks with PreparedChatCompletion
    done_cb = _task_done_callback_factory(state.inflight_request_tasks)

    for i, pending in enumerate(batch):
        if pending.future.done():
            continue

        backend = assigned_by_i[i]
        if backend is None:
            try_set_future_result(
                pending.future,
                (503, {"error": {"message": "Routing failed: no backend assigned"}}),
            )
            continue

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

        scores = backend_scores_by_req[i]
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

        chosen_score = _find_chosen_backend_score(scores, backend_id=backend.backend_id)
        if chosen_score is None:
            try_set_future_result(
                pending.future,
                (
                    500,
                    {
                        "error": {
                            "message": "Internal router error: missing chosen score"
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
            backend_scores=list(scores),
            chosen_predictor_input=chosen_inp,
            batch_id=int(info.batch_id),
            batch_size=int(info.batch_size),
            routing_policy=str(state.routing_policy),
            auction_matched=bool(auction_matched_by_i[i]),
            auction_total_welfare=float(auction_total_welfare),
            chosen_client_valuation=float(chosen_score.client_valuation),
            chosen_base_cost=float(chosen_score.base_cost),
            chosen_welfare=float(chosen_score.welfare),
            vcg_fee=vcg_fee_by_i[i] if auction_matched_by_i[i] else None,
            vcg_total_payment=vcg_pay_by_i[i] if auction_matched_by_i[i] else None,
        )

        t = asyncio.create_task(
            _process_one_chat_completion(prep, state=state),
            name=f"chat_completion.batch{info.batch_id}.i{i}",
        )
        state.inflight_request_tasks.add(t)
        t.add_done_callback(done_cb)
