"""
immas.router.warmup

Internal startup warmup for the router.

This module performs a small number of synthetic chat-completion requests to
each configured backend during router startup (FastAPI lifespan), to:

- avoid first-request backend anomalies (model load, compilation, etc.)
- bootstrap online predictors so auction routing does not degenerate at cold start

Warmup is intentionally *not* exposed to clients and does not write JSONL logs.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import Any, Optional

from immas.openai.chat import serialize_chat_messages
from immas.openai.usage import parse_usage
from immas.router.components.predictor import PredictorInput
from immas.router.state import RouterState

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WarmupOutcome:
    """One warmup request outcome."""

    backend_id: str
    ok: bool
    obs_latency_ms: float
    obs_total_tokens: int
    error: Optional[str] = None


def _make_warmup_messages(*, backend_id: str, k: int) -> list[dict[str, Any]]:
    """
    Create a small deterministic warmup conversation.

    Keep it short to avoid polluting backend KV cache or consuming significant compute.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Warmup request {k} for backend {backend_id}. Reply with 'ok'.",
        },
    ]


async def warmup_router(state: RouterState) -> None:
    """
    Run startup warmup according to config.

    Notes
    -----
    - This runs before the micro-batcher starts and before the app begins serving traffic.
    - Failures are logged but do not prevent the router from starting.
    """
    cfg = state.cfg.router.warmup
    if not cfg.enabled:
        _log.info("Warmup disabled (router.warmup.enabled=false)")
        return
    if cfg.requests_per_backend <= 0:
        _log.info("Warmup enabled but requests_per_backend <= 0; skipping")
        return
    if not state.backends:
        _log.warning("Warmup skipped: no backends configured")
        return

    _log.info(
        "Warmup starting: backends=%s requests_per_backend=%s max_concurrency=%s timeout_s=%s",
        len(state.backends),
        cfg.requests_per_backend,
        cfg.max_concurrency,
        cfg.timeout_s,
    )

    sem = asyncio.Semaphore(int(cfg.max_concurrency))
    outcomes: list[WarmupOutcome] = []

    async def _one(backend_idx: int, k: int) -> None:
        backend = state.backends[backend_idx]
        backend_id = backend.backend_id
        model = state.backend_model_by_id.get(backend_id, "")
        if not model:
            outcomes.append(
                WarmupOutcome(
                    backend_id=backend_id,
                    ok=False,
                    obs_latency_ms=0.0,
                    obs_total_tokens=0,
                    error="missing_backend_model",
                )
            )
            return

        messages = _make_warmup_messages(backend_id=backend_id, k=k)
        prompt_repr = serialize_chat_messages(messages)

        # Snapshot load (very likely zeros at startup, but keep it correct).
        router_snap = await state.load_tracker.snapshot()
        backend_tracker = state.backend_load_trackers.get(backend_id)
        if backend_tracker is None:
            backend_inflight = 0
            backend_rps = 0.0
        else:
            bs = await backend_tracker.snapshot()
            backend_inflight = int(bs.inflight_requests)
            backend_rps = float(bs.rps)

        cap = max(1, int(state.backend_capacity_by_id.get(backend_id, 1)))

        inp = PredictorInput(
            backend_id=backend_id,
            model=model,
            source="warmup",
            dialogue_id="__immas_warmup__",
            turn_number=int(k),
            prompt_repr=prompt_repr,
            kvmatch_text=0.0,
            router_inflight=int(router_snap.inflight_requests),
            router_rps_1s=float(router_snap.rps),
            backend_inflight=int(backend_inflight),
            backend_rps_1s=float(backend_rps),
            backend_capacity=int(cap),
        )

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(cfg.max_tokens),
            "stream": False,
        }

        async with sem:
            t0 = time.monotonic()
            try:
                backend_sem = state.backend_semaphores.get(
                    backend_id
                ) or asyncio.Semaphore(1)

                async with backend_sem:
                    # Track inflight for both global and backend-local load trackers.
                    if backend_tracker is None:
                        async with state.load_tracker.track():
                            status, resp_json = await asyncio.wait_for(
                                backend.forward_chat_completions(body, headers=None),
                                timeout=float(cfg.timeout_s),
                            )
                    else:
                        async with state.load_tracker.track(), backend_tracker.track():
                            status, resp_json = await asyncio.wait_for(
                                backend.forward_chat_completions(body, headers=None),
                                timeout=float(cfg.timeout_s),
                            )

                t1 = time.monotonic()
                obs_latency_ms = (t1 - t0) * 1000.0

                if not (200 <= int(status) < 300):
                    outcomes.append(
                        WarmupOutcome(
                            backend_id=backend_id,
                            ok=False,
                            obs_latency_ms=float(obs_latency_ms),
                            obs_total_tokens=0,
                            error=f"backend_status={status}",
                        )
                    )
                    return

                usage = parse_usage(resp_json if isinstance(resp_json, dict) else {})
                obs_total_tokens = int(usage.total_tokens)

                # Update predictor (no logging).
                await state.predictors.update_one(
                    inp,
                    real_latency_ms=float(obs_latency_ms),
                    real_cost_tokens=int(obs_total_tokens),
                    real_perf_correct=True,
                )

                outcomes.append(
                    WarmupOutcome(
                        backend_id=backend_id,
                        ok=True,
                        obs_latency_ms=float(obs_latency_ms),
                        obs_total_tokens=int(obs_total_tokens),
                    )
                )
            except asyncio.TimeoutError:
                outcomes.append(
                    WarmupOutcome(
                        backend_id=backend_id,
                        ok=False,
                        obs_latency_ms=0.0,
                        obs_total_tokens=0,
                        error="timeout",
                    )
                )
            except Exception as e:
                outcomes.append(
                    WarmupOutcome(
                        backend_id=backend_id,
                        ok=False,
                        obs_latency_ms=0.0,
                        obs_total_tokens=0,
                        error=f"{type(e).__name__}: {e}",
                    )
                )

    tasks: list[asyncio.Task[None]] = []
    for bi in range(len(state.backends)):
        for k in range(1, int(cfg.requests_per_backend) + 1):
            tasks.append(asyncio.create_task(_one(bi, k)))

    await asyncio.gather(*tasks, return_exceptions=False)

    # Summarize.
    ok = [o for o in outcomes if o.ok]
    err = [o for o in outcomes if not o.ok]

    by_backend: dict[str, list[WarmupOutcome]] = {}
    for o in outcomes:
        by_backend.setdefault(o.backend_id, []).append(o)

    parts: list[str] = []
    for bid, outs in sorted(by_backend.items()):
        oks = [x for x in outs if x.ok]
        if oks:
            avg_lat = sum(x.obs_latency_ms for x in oks) / len(oks)
            avg_tok = sum(x.obs_total_tokens for x in oks) / len(oks)
            parts.append(
                f"{bid}: ok={len(oks)}/{len(outs)} avg_lat_ms={avg_lat:.1f} avg_tok={
                    avg_tok:.1f}"
            )
        else:
            parts.append(f"{bid}: ok=0/{len(outs)}")

    _log.info(
        "Warmup finished: ok=%s err=%s (%s)",
        len(ok),
        len(err),
        "; ".join(parts),
    )
    if err:
        # Log a few representative errors (avoid huge logs).
        for o in err[: min(10, len(err))]:
            _log.warning("Warmup error backend=%s err=%s", o.backend_id, o.error)
