"""
Fake OpenAI-compatible server for CoQA.

Implements:
- GET  /v1/models
- POST /v1/chat/completions

The server answers by looking up the gold CoQA answer using (dialogue_id, turn_number)
parsed from the prompt text.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from coqa_data import CoqaDatasetIndex
from coqa_prompt import CoqaPromptParser


DEFAULT_MODEL_NAME = "fake-coqa"


def estimate_tokens(text: str) -> int:
    """
    Crude token estimator used only for 'usage' fields.

    1 token ~= 4 characters is a common rough heuristic.
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
    app = FastAPI(title="Fake CoQA OpenAI API", version="0.1")

    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", DEFAULT_MODEL_NAME)

    oracle: Optional[CoqaAnswerOracle] = None

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal oracle
        oracle = CoqaAnswerOracle(split=split)

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

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> Dict[str, Any]:
        nonlocal oracle
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

        # Usage counts across *all* messages to roughly mimic OpenAI usage accounting.
        prompt_tokens = sum(
            estimate_tokens(str(m.get("content") or ""))
            for m in messages
            if isinstance(m, dict)
        )
        completion_tokens = estimate_tokens(answer)

        now = int(time.time())
        resp_model = str(body.get("model") or model_name)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
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
