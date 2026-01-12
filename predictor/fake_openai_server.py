"""
Fake OpenAI-compatible server for CoQA with vLLM-ish load/latency simulation.

Implements:
- GET  /v1/models
- GET  /v1/internal/load          (debug)
- POST /v1/chat/completions

The server answers by looking up the gold CoQA answer using (dialogue_id, turn_number)
parsed from the prompt text.

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
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from coqa_data import CoqaDatasetIndex
from coqa_prompt import CoqaPromptParser
from load_tracker import AsyncLoadTracker
from trace_store import ChatCompletionTrace, InMemoryTraceStore
from vllm_latency_sim import VllmLatencySimConfig, VllmLatencySimulator


DEFAULT_MODEL_NAME = "fake-coqa"


def estimate_tokens(text: str) -> int:
    """
    Crude token estimator used only for 'usage' fields and simulation.

    A common heuristic: 1 token ~= 4 characters.
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


class CoqaAnswerOracle:
    """Dataset-backed oracle that returns gold answers."""

    def __init__(self, *, split: str) -> None:
        self._index = CoqaDatasetIndex.from_hf(split=split)

    def answer_from_prompt(self, prompt: str) -> str:
        parsed = CoqaPromptParser.parse(prompt)
        return self._index.get_answer(
            dialogue_id=parsed.dialogue_id, turn_number=parsed.turn_number
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Fake CoQA OpenAI API", version="0.2")

    app.state.trace_store = InMemoryTraceStore(
        max_size=int(os.environ.get("FAKE_TRACE_STORE_MAX", "50000"))
    )

    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", DEFAULT_MODEL_NAME)

    # Create these early; they are lightweight and per-process.
    app.state.load_tracker = AsyncLoadTracker(
        window_s=float(os.environ.get("FAKE_LOAD_WINDOW_S", "1.0"))
    )
    app.state.lat_sim = VllmLatencySimulator(VllmLatencySimConfig.from_env())

    @app.on_event("startup")
    def _startup() -> None:
        app.state.oracle = CoqaAnswerOracle(split=split)

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

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> Dict[str, Any]:
        oracle = getattr(req.app.state, "oracle", None)
        if oracle is None:
            raise HTTPException(
                status_code=503, detail="Server not ready (oracle not loaded yet)."
            )

        body = await req.json()

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return openai_error("Missing or invalid 'messages' list.", status_code=400)

        # Find the last user message content; fallback to last message.
        prompt_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = str(msg.get("content") or "")
                break
        if not prompt_text and isinstance(messages[-1], dict):
            prompt_text = str(messages[-1].get("content") or "")

        if not prompt_text:
            return openai_error(
                "Could not extract prompt text from messages.", status_code=400
            )

        try:
            answer = oracle.answer_from_prompt(prompt_text)
        except KeyError as e:
            return openai_error(f"Dialogue not found: {e}", status_code=404)
        except (ValueError, IndexError) as e:
            return openai_error(f"Prompt parse/turn error: {e}", status_code=400)

        # Usage counts across all messages to mimic OpenAI usage accounting.
        prompt_tokens = sum(
            estimate_tokens(str(m.get("content") or ""))
            for m in messages
            if isinstance(m, dict)
        )
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
