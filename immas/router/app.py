"""
immas.router.app

A lightweight OpenAI-compatible router that:
- receives /v1/chat/completions
- computes prediction-time features (including text-based KV match proxy)
- selects a backend (single backend; routing is pluggable)
- forwards the request to the backend
- logs + online-trains a predictor based on observed outcomes

-------------
- We measure observed latency as end-to-end (E2E) wall time at the router.
- Correctness is set to True (placeholder).
- The backend is expected to speak OpenAI-compatible HTTP+JSON.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from immas.common.load import AsyncLoadTracker
from immas.common.text import extract_last_user_text, inline_for_cache
from immas.router.backend import HttpOpenAIBackend, OpenAIBackend
from immas.router.logger import AsyncJsonlLogger, RouterLogRecord
from immas.router.predictor import AgentPredictor, PredictorInput
from immas.router.prefix_cache import TextPrefixCache


_HEADER_RUN_ID = "x-immas-run-id"
_HEADER_DIALOGUE_ID = "x-immas-dialogue-id"
_HEADER_TURN_NUMBER = "x-immas-turn-number"
_HEADER_SOURCE = "x-immas-source"


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Runtime configuration for the router."""

    backend_id: str
    backend_base_url_v1: str

    log_path: str
    log_append: bool


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_config() -> RouterConfig:
    """
    Phase 0: single backend configured by env.
    Later: extend to multiple backends.
    """
    backend_base_url_v1 = os.environ.get(
        "IMMAS_BACKEND_BASE_URL", "http://localhost:8000/v1"
    ).strip()
    backend_id = os.environ.get("IMMAS_BACKEND_ID", "b0").strip()

    log_path = os.environ.get("IMMAS_ROUTER_LOG_PATH", "router_run.jsonl")
    log_append = _env_bool("IMMAS_ROUTER_LOG_APPEND", False)

    return RouterConfig(
        backend_id=backend_id,
        backend_base_url_v1=backend_base_url_v1,
        log_path=log_path,
        log_append=log_append,
    )


def _get_required_header(req: Request, name: str) -> str:
    v = req.headers.get(name)
    return (v or "").strip()


def _parse_turn_number(raw: str) -> int:
    try:
        n = int(raw)
        return n if n >= 0 else 0
    except Exception:
        return 0


def create_app() -> FastAPI:
    cfg = _load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Predictor state (online learning) - protect with lock.
        app.state.predictor = AgentPredictor()
        app.state.predictor_lock = asyncio.Lock()

        # Router feature cache (text prefix) - protect with lock.
        app.state.prefix_cache = TextPrefixCache()
        app.state.prefix_cache_lock = asyncio.Lock()

        # Router-side load tracker (inflight + RPS).
        app.state.load_tracker = AsyncLoadTracker(window_s=1.0)

        # Backend client.
        app.state.backend = HttpOpenAIBackend(
            backend_id=cfg.backend_id,
            base_url_v1=cfg.backend_base_url_v1,
        )

        # JSONL logger
        app.state.logger = AsyncJsonlLogger(
            cfg.log_path, append=cfg.log_append, flush_every=1
        )
        await app.state.logger.__aenter__()

        yield

        # Shutdown
        await app.state.logger.close()
        await app.state.backend.close()

    app = FastAPI(title="IMMAS Router", version="0.1", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models(req: Request) -> JSONResponse:
        backend: OpenAIBackend = req.app.state.backend
        status, payload = await backend.list_models()
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: Request) -> JSONResponse:
        backend: OpenAIBackend = req.app.state.backend
        predictor: AgentPredictor = req.app.state.predictor
        predictor_lock: asyncio.Lock = req.app.state.predictor_lock
        cache: TextPrefixCache = req.app.state.prefix_cache
        cache_lock: asyncio.Lock = req.app.state.prefix_cache_lock
        load_tracker: AsyncLoadTracker = req.app.state.load_tracker
        logger: AsyncJsonlLogger = req.app.state.logger

        run_id = _get_required_header(req, _HEADER_RUN_ID) or "run_unknown"
        dialogue_id = (
            _get_required_header(req, _HEADER_DIALOGUE_ID) or "dialogue_unknown"
        )
        turn_number = _parse_turn_number(_get_required_header(req, _HEADER_TURN_NUMBER))
        source = _get_required_header(req, _HEADER_SOURCE) or "unknown"

        body = await req.json()

        # OpenAI schema: body["model"], body["messages"] are expected.
        model = str(body.get("model") or "")
        messages = body.get("messages")

        prompt_text = extract_last_user_text(messages)
        prompt_chars = len(prompt_text)

        # Compute kvmatch against router cache
        async with cache_lock:
            kvmatch = cache.match_ratio(
                backend_id=backend.backend_id,
                model=model,
                dialogue_id=dialogue_id,
                prompt_text=prompt_text,
            )

        async with load_tracker.track() as load:
            inp = PredictorInput(
                model=model,
                source=source,
                dialogue_id=dialogue_id,
                turn_number=turn_number,
                prompt_text=prompt_text,
                kvmatch=float(kvmatch),
                router_inflight=int(load.inflight_requests),
                router_rps_1s=float(load.rps),
            )

            # Predict
            async with predictor_lock:
                pred = predictor.predict(inp)

            pred_latency_ms = float(pred["latency_ms"][0])
            pred_cost_tokens = float(pred["cost_tokens"][0])
            pred_perf_prob = float(pred["performance"][0])

            # Forward to backend + observe
            t0 = time.perf_counter()
            status, resp_json = await backend.forward_chat_completions(body)
            t1 = time.perf_counter()

        obs_latency_ms = (t1 - t0) * 1000.0

        # Extract observed tokens + assistant answer (best-effort).
        completion_id = str(resp_json.get("id") or "")
        usage = resp_json.get("usage") if isinstance(resp_json, dict) else None
        obs_total_tokens = 0
        if isinstance(usage, dict):
            try:
                obs_total_tokens = int(usage.get("total_tokens") or 0)
            except Exception:
                obs_total_tokens = 0

        answer_text = ""
        try:
            if isinstance(resp_json, dict):
                choices = resp_json.get("choices")
                if isinstance(choices, list) and choices:
                    msg = (
                        choices[0].get("message")
                        if isinstance(choices[0], dict)
                        else None
                    )
                    if isinstance(msg, dict):
                        answer_text = str(msg.get("content") or "")
        except Exception:
            answer_text = ""

        # FIXME: Placeholder correctness: always True.
        correct = True

        # Online update.
        error: Optional[str] = None
        if 200 <= status < 300:
            async with predictor_lock:
                predictor.update(
                    inp,
                    real_latency_ms=float(obs_latency_ms),
                    real_cost_tokens=int(obs_total_tokens),
                    real_perf_correct=bool(correct),
                )

            # Update router cache so next turn gets a higher prefix match.
            # We mimic the client's prompt evolution: next prompt contains prompt + " " + inline(answer).
            if prompt_text:
                async with cache_lock:
                    cache.update(
                        backend_id=backend.backend_id,
                        model=model,
                        dialogue_id=dialogue_id,
                        cached_text=prompt_text + " " + inline_for_cache(answer_text),
                    )
        else:
            error = f"backend_status={status}"

        rec = RouterLogRecord(
            run_id=run_id,
            t_start_monotonic=float(t0),
            t_end_monotonic=float(t1),
            backend_id=backend.backend_id,
            backend_base_url_v1=backend.base_url_v1,
            model=model,
            source=source,
            dialogue_id=dialogue_id,
            turn_number=int(turn_number),
            prompt_chars=int(prompt_chars),
            kvmatch=float(kvmatch),
            router_inflight=int(inp.router_inflight),
            router_rps_1s=float(inp.router_rps_1s),
            pred_latency_ms=float(pred_latency_ms),
            pred_cost_tokens=float(pred_cost_tokens),
            pred_perf_prob=float(pred_perf_prob),
            completion_id=completion_id,
            obs_latency_ms=float(obs_latency_ms),
            obs_total_tokens=int(obs_total_tokens),
            correct=bool(correct),
            error=error,
        )
        await logger.log(rec)

        return JSONResponse(status_code=status, content=resp_json)

    return app


app = create_app()
