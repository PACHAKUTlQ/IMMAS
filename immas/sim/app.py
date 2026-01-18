"""
immas.sim.app

Fake OpenAI-compatible server for CoQA with vLLM-ish load/latency simulation.

Implements:
- GET  /v1/models
- GET  /v1/internal/load          (debug)
- GET  /v1/internal/chat_completions/{completion_id}  (debug trace fetch)
- POST /v1/chat/completions

Behavior
--------
This fake server returns the gold CoQA answer for each turn.

There are two supported ways to determine which (dialogue_id, turn_number)
the request refers to:

1) Header-based (preferred):
   - X-IMMAS-DIALOGUE-ID
   - X-IMMAS-TURN-NUMBER

   This path is robust even when the request body is a normal multi-turn
   OpenAI chat `messages` list (system/story/questions/assistant history).

2) Prompt-parsing fallback (legacy):
   If the headers are missing, we attempt to parse the last user message as a
   deterministic prompt produced by `CoqaPromptFormatter`, using `CoqaPromptParser`.
   This is what was previously meant by "parse from prompt".

Latency simulation
------------------
- Tracks in-flight and recent RPS per process.
- Simulates TTFT + decode latency with occasional batch-correlated spikes.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Union, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from immas.data.coqa.loader import CoqaDatasetIndex
from immas.data.coqa.prompt import CoqaPromptParser
from immas.sim.latency import VllmLatencySimConfig, VllmLatencySimulator
from immas.sim.load import AsyncLoadTracker
from immas.sim.traces import ChatCompletionTrace, InMemoryTraceStore


DEFAULT_MODEL_NAME = "fake-coqa"

_HEADER_DIALOGUE_ID = "x-immas-dialogue-id"
_HEADER_TURN_NUMBER = "x-immas-turn-number"


def estimate_tokens(text: str) -> int:
    """
    Crude token estimator used only for 'usage' fields and simulation.

    Heuristic: 1 token is approximately 4 characters.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def openai_error(message: str, *, status_code: int = 400) -> JSONResponse:
    """Return an OpenAI-ish error envelope."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def _extract_last_user_text(messages: Any) -> str:
    """
    Extract the last user message text from a chat 'messages' array.

    This is used only to:
    - optionally parse legacy CoQA prompt formatting, and
    - provide a stable prompt_text for debug/estimation.
    """
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    if messages and isinstance(messages[-1], dict):
        return str(messages[-1].get("content") or "")
    return ""


def _parse_turn_number_header(raw: str) -> int:
    try:
        n = int(raw)
        return n if n >= 1 else 0
    except Exception:
        return 0


class CoqaAnswerOracle:
    """
    Dataset-backed oracle that returns gold answers.

    We intentionally support both:
    - header-based lookup (dialogue_id, turn_number), and
    - legacy prompt parsing fallback.
    """

    def __init__(self, *, split: str) -> None:
        self._index = CoqaDatasetIndex.from_hf(split=split)

    def answer_by_id_turn(self, *, dialogue_id: str, turn_number: int) -> str:
        """Return gold answer for a given (dialogue_id, turn_number)."""
        return self._index.get_answer(dialogue_id=dialogue_id, turn_number=turn_number)

    def answer_from_prompt(self, prompt: str) -> str:
        """
        Legacy path: parse (dialogue_id, turn_number) from a formatted prompt and answer.
        """
        parsed = CoqaPromptParser.parse(prompt)
        return self._index.get_answer(
            dialogue_id=parsed.dialogue_id, turn_number=parsed.turn_number
        )


def create_app() -> FastAPI:
    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", DEFAULT_MODEL_NAME)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup:
        app.state.oracle = CoqaAnswerOracle(split=split)
        yield
        # Shutdown: no-op.

    app = FastAPI(title="Fake CoQA OpenAI API", version="0.3", lifespan=lifespan)

    # In-memory debug trace store (per-process).
    app.state.trace_store = InMemoryTraceStore(
        max_size=int(os.environ.get("FAKE_TRACE_STORE_MAX", "50000"))
    )

    # Per-process load tracker + latency simulator.
    app.state.load_tracker = AsyncLoadTracker(
        window_s=float(os.environ.get("FAKE_LOAD_WINDOW_S", "1.0"))
    )
    app.state.lat_sim = VllmLatencySimulator(VllmLatencySimConfig.from_env())

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "fake",
                }
            ],
        }

    @app.get("/v1/internal/chat_completions/{completion_id}")
    async def get_chat_completion_trace(
        completion_id: str, req: Request
    ) -> Dict[str, Any]:
        store: InMemoryTraceStore = req.app.state.trace_store
        trace = await store.get(completion_id)
        if trace is None:
            raise HTTPException(
                status_code=404, detail="Trace not found (evicted or wrong worker)."
            )
        return trace.to_dict()

    @app.get("/v1/internal/load")
    async def internal_load(req: Request) -> Dict[str, Any]:
        """
        Debug endpoint to inspect server-side load.
        Not part of the OpenAI API; safe to ignore.
        """
        tracker: AsyncLoadTracker = req.app.state.load_tracker
        snap = await tracker.snapshot()
        return {
            "inflight_requests": snap.inflight_requests,
            "rps": snap.rps,
            "window_s": snap.window_s,
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: Request) -> Union[Dict[str, Any], JSONResponse]:
        oracle = getattr(req.app.state, "oracle", None)
        if oracle is None:
            raise HTTPException(
                status_code=503, detail="Server not ready (oracle not loaded yet)."
            )
        oracle = cast(CoqaAnswerOracle, oracle)

        body = await req.json()
        if not isinstance(body, dict):
            return openai_error(
                "Invalid JSON body; expected an object.", status_code=400
            )

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return openai_error("Missing or invalid 'messages' list.", status_code=400)

        # Extract a prompt_text for legacy parsing and for token estimation.
        prompt_text = _extract_last_user_text(messages)
        if not prompt_text:
            return openai_error(
                "Could not extract prompt text from messages.", status_code=400
            )

        did = (req.headers.get(_HEADER_DIALOGUE_ID) or "").strip()
        turn_number = _parse_turn_number_header(
            req.headers.get(_HEADER_TURN_NUMBER) or ""
        )

        try:
            if did and turn_number >= 1:
                answer = oracle.answer_by_id_turn(
                    dialogue_id=did, turn_number=turn_number
                )
            else:
                # Legacy fallback:
                # Parse dialogue_id and turn_number from a deterministic prompt text.
                answer = oracle.answer_from_prompt(prompt_text)
        except KeyError as e:
            return openai_error(f"Dialogue not found: {e}", status_code=404)
        except (ValueError, IndexError) as e:
            return openai_error(f"Prompt parse/turn error: {e}", status_code=400)

        # Usage counts across all messages to mimic OpenAI usage accounting.
        prompt_tokens = 0
        for m in messages:
            if isinstance(m, dict):
                prompt_tokens += estimate_tokens(str(m.get("content") or ""))
        completion_tokens = estimate_tokens(answer)

        # --- Simulated latency (async, non-blocking) ---
        tracker: AsyncLoadTracker = req.app.state.load_tracker
        lat_sim: VllmLatencySimulator = req.app.state.lat_sim

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        resp_model = str(body.get("model") or model_name)
        t_server0 = time.perf_counter()

        async with tracker.track() as load:
            sim = lat_sim.simulate(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                load=load,
            )
            if sim.total_s > 0:
                await asyncio.sleep(sim.total_s)

        t_server1 = time.perf_counter()
        server_wall_s = float(t_server1 - t_server0)

        store: InMemoryTraceStore = req.app.state.trace_store
        await store.put(
            ChatCompletionTrace(
                completion_id=completion_id,
                created=created,
                model=resp_model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
                load_inflight=int(load.inflight_requests),
                load_rps=float(load.rps),
                sim_ttft_s=float(sim.ttft_s),
                sim_decode_s=float(sim.decode_s),
                sim_queue_s=float(sim.queue_s),
                sim_stall_s=float(sim.stall_s),
                sim_warmup_s=float(sim.warmup_s),
                sim_total_s=float(sim.total_s),
                effective_inflight=int(sim.effective_inflight),
                utilization=float(sim.utilization),
                sim_rps=float(sim.rps),
                server_wall_s=server_wall_s,
            )
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": resp_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
