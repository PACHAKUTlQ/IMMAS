"""
immas.router.app

A lightweight OpenAI-compatible router that:
- receives /v1/chat/completions
- enqueues requests into a micro-batcher
- micro-batcher forms short batches with strict max-wait bound
- routes each request (currently: round-robin) and forwards to backends
- logs + online-trains predictors based on observed outcomes
- uses backend API keys from its own YAML configuration (no client auth passthrough)
"""

from __future__ import annotations

import asyncio
import logging
import time

from functools import partial
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from immas.router.lifecycle import lifespan
from immas.router.state import RouterState
from immas.router.types import ChatCompletionResult, PendingChatCompletion
from immas.router.utils import (
    _get_header,
    _parse_turn_number,
    load_cfg_from_env,
)


_log = logging.getLogger(__name__)

_HEADER_RUN_ID = "x-immas-run-id"
_HEADER_DIALOGUE_ID = "x-immas-dialogue-id"
_HEADER_TURN_NUMBER = "x-immas-turn-number"
_HEADER_SOURCE = "x-immas-source"


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    cfg = load_cfg_from_env()
    app_lifespan = partial(lifespan, cfg=cfg)
    app = FastAPI(title="IMMAS Router", version="0.5", lifespan=app_lifespan)

    @app.get("/v1/models")
    async def list_models(req: Request) -> JSONResponse:
        """
        Return the union of router-configured backend model names.
        """

        router_state: RouterState = req.app.state.router_state
        seen: set[str] = set()
        data: list[dict[str, Any]] = []
        for m in router_state.backend_model_by_id.values():
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

        router_state: RouterState = req.app.state.router_state
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

        ok = router_state.chat_batcher.try_submit(pending)
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Router overloaded (batch queue full)"}},
            )

        status, resp_json = await fut
        return JSONResponse(status_code=int(status), content=resp_json)

    return app


app = create_app()
