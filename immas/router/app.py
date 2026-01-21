"""
immas.router.app

A lightweight OpenAI-compatible router that:
- receives /v1/chat/completions
- computes router-side features (including text-based KV match proxy)
- selects a backend (currently: round-robin)
- forwards the request to the chosen backend
- logs + online-trains a predictor based on observed outcomes
- uses backend API keys from its own YAML configuration.

Cache ratio prediction
----------------------
We treat `pred_cache_ratio := kvmatch_text` and feed that deterministic value into the latency/cost/perf models.

Router-side prefix cache eviction
---------------------------------
Some backends (e.g. vLLM) may evict prompt-cache entries. When the backend reports
cached-token accounting and we observe a near-zero cache ratio despite a near-
perfect prefix match, we conservatively evict the router's corresponding prefix
cache record and skip updating it for that request.
"""

from __future__ import annotations

import asyncio
import os
import time

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from immas.common.load import AsyncLoadTracker
from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import ParsedUsage, parse_usage
from immas.router.backend import HttpOpenAIBackend
from immas.router.config import RouterAppConfig, load_router_app_config
from immas.router.logger import AsyncJsonlLogger, RouterLogRecord
from immas.router.predictor import AgentPredictor, PredictorInput
from immas.router.prefix_cache import TextPrefixCache


_HEADER_RUN_ID = "x-immas-run-id"
_HEADER_DIALOGUE_ID = "x-immas-dialogue-id"
_HEADER_TURN_NUMBER = "x-immas-turn-number"
_HEADER_SOURCE = "x-immas-source"

# Conservative eviction heuristic thresholds.
_EVICT_KVMATCH_MIN = 0.8
_EVICT_OBS_CACHE_MAX = 0.10
_EVICT_MIN_PROMPT_TOKENS = 64


def _get_header(req: Request, name: str) -> str:
    return (req.headers.get(name) or "").strip()


def _parse_turn_number(raw: str) -> int:
    try:
        n = int(raw)

        return n if n >= 0 else 0
    except Exception:
        return 0


def _load_cfg_from_env() -> RouterAppConfig:
    """
    Load router config from YAML path in env IMMAS_ROUTER_CONFIG.

    This is the primary configuration mechanism going forward.
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


def create_app() -> FastAPI:
    cfg = _load_cfg_from_env()

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

        # Round-robin state.
        app.state.rr_lock = asyncio.Lock()
        app.state.rr_index = 0

        # JSONL logger
        app.state.logger = AsyncJsonlLogger(
            cfg.router.log_path, append=cfg.router.log_append, flush_every=1
        )
        await app.state.logger.__aenter__()

        yield

        # Shutdown
        await app.state.logger.close()
        for b in app.state.backends:
            await b.close()

    app = FastAPI(title="IMMAS Router", version="0.4", lifespan=lifespan)

    async def _select_backend(req: Request) -> HttpOpenAIBackend:
        """
        Select one backend for this request.

        Current policy: round-robin over configured backends.
        """

        backends: list[HttpOpenAIBackend] = req.app.state.backends
        rr_lock: asyncio.Lock = req.app.state.rr_lock

        async with rr_lock:
            idx: int = int(req.app.state.rr_index)
            req.app.state.rr_index = (idx + 1) % len(backends)

        return backends[idx]

    @app.get("/v1/models")
    async def list_models(req: Request) -> JSONResponse:
        """
        Return the union of router-configured backend model names.

        Rationale: in this system, the router is the "source of truth" for models
        because backends may expose different model names and the client should
        not choose backend models directly.
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
        backend = await _select_backend(req)
        backend_model_by_id: dict[str, str] = req.app.state.backend_model_by_id
        backend_model = backend_model_by_id.get(backend.backend_id, "")

        predictor: AgentPredictor = req.app.state.predictor
        predictor_lock: asyncio.Lock = req.app.state.predictor_lock
        cache: TextPrefixCache = req.app.state.prefix_cache
        cache_lock: asyncio.Lock = req.app.state.prefix_cache_lock
        load_tracker: AsyncLoadTracker = req.app.state.load_tracker
        logger: AsyncJsonlLogger = req.app.state.logger

        run_id = _get_header(req, _HEADER_RUN_ID) or "run_unknown"
        dialogue_id = _get_header(req, _HEADER_DIALOGUE_ID) or "dialogue_unknown"
        turn_number = _parse_turn_number(_get_header(req, _HEADER_TURN_NUMBER))
        source = _get_header(req, _HEADER_SOURCE) or "unknown"

        body: Any = await req.json()
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"error": {"message": "Invalid JSON body"}}
            )

        # Client-provided model is ignored; router enforces backend-specific model.
        if not backend_model:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"No configured model for backend {backend.backend_id}"
                    }
                },
            )

        # Use *effective* model for feature computation / caching / logging.
        effective_model = backend_model

        messages = body.get("messages")
        prompt_repr = serialize_chat_messages(messages)
        prompt_chars = len(prompt_repr)

        # KV-match proxy against router-side cache *for the chosen backend*.
        async with cache_lock:
            pm = cache.match(
                backend_id=backend.backend_id,
                model=effective_model,
                dialogue_id=dialogue_id,
                prompt_text=prompt_repr,
            )
        kvmatch_text = float(pm.ratio)
        cached_prompt_chars = int(pm.cached_chars)
        kvmatch_lcp_chars = int(pm.lcp_chars)

        async with load_tracker.track() as load:
            inp = PredictorInput(
                backend_id=backend.backend_id,
                model=effective_model,
                source=source,
                dialogue_id=dialogue_id,
                turn_number=turn_number,
                prompt_repr=prompt_repr,
                kvmatch_text=float(kvmatch_text),
                router_inflight=int(load.inflight_requests),
                router_rps_1s=float(load.rps),
            )

            # Predict
            async with predictor_lock:
                pred = predictor.predict(inp)

            pred_latency_ms = float(pred["latency_ms"][0])
            pred_cost_tokens = float(pred["cost_tokens"][0])
            pred_perf_prob = float(pred["performance"][0])
            # deterministic == kvmatch_text
            pred_cache_ratio = float(pred["cache_ratio"][0])

            # Router-controlled headers to backend (no client auth passthrough).
            backend_headers = {
                "X-IMMAS-RUN-ID": run_id,
                "X-IMMAS-DIALOGUE-ID": dialogue_id,
                "X-IMMAS-TURN-NUMBER": str(turn_number),
                "X-IMMAS-SOURCE": source,
            }

            # Forward a copy with router-enforced model.
            forwarded_body: dict[str, Any] = dict(body)
            forwarded_body["model"] = effective_model

            t0 = time.perf_counter()
            status, resp_json = await backend.forward_chat_completions(
                forwarded_body,
                headers=backend_headers,
            )
            t1 = time.perf_counter()

        obs_latency_ms = (t1 - t0) * 1000.0

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
            turn_number=int(turn_number),
            kvmatch_text=float(kvmatch_text),
            obs_cache_ratio=float(obs_cache_ratio),
        )

        # Update predictor + cache only on success
        if 200 <= status < 300:
            async with predictor_lock:
                predictor.update(
                    inp,
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
                        dialogue_id=dialogue_id,
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
                            dialogue_id=dialogue_id,
                            cached_text=new_prompt_repr,
                        )
        else:
            error = f"backend_status={status}"

        rec = RouterLogRecord(
            run_id=run_id,
            t_start_monotonic=float(t0),
            t_end_monotonic=float(t1),
            backend_id=backend.backend_id,
            backend_base_url_v1=backend.base_url_v1,
            model=effective_model,
            source=source,
            dialogue_id=dialogue_id,
            turn_number=int(turn_number),
            prompt_chars=int(prompt_chars),
            cached_prompt_chars=int(cached_prompt_chars),
            kvmatch_lcp_chars=int(kvmatch_lcp_chars),
            kvmatch_text=float(kvmatch_text),
            router_inflight=int(inp.router_inflight),
            router_rps_1s=float(inp.router_rps_1s),
            pred_latency_ms=float(pred_latency_ms),
            pred_cost_tokens=float(pred_cost_tokens),
            pred_perf_prob=float(pred_perf_prob),
            pred_cache_ratio=float(pred_cache_ratio),
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

        return JSONResponse(status_code=status, content=resp_json)

    return app


app = create_app()
