"""
immas.router.components.backend

Backend abstraction for OpenAI-compatible servers.
Uses a simple HTTP JSON forwarder.
Router uses backend API keys from its own configuration.

Notes
-----
This router optionally forces upstream streaming for /chat/completions in order to
measure a TTFT-like proxy latency (time to first meaningful stream chunk) while
still returning a standard non-streaming JSON response to clients.

We embed the measured timestamp in the reconstructed JSON payload under a private
key and the router strips it before returning to the client.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Tuple

import httpx


_IMMAS_T_FIRST_TOKEN_MONOTONIC_KEY = "_immas_t_first_token_monotonic"


class OpenAIBackend(Protocol):
    """Protocol for router backends."""

    @property
    def backend_id(self) -> str: ...

    @property
    def base_url_v1(self) -> str: ...

    async def list_models(self) -> Tuple[int, Dict[str, Any]]: ...

    async def forward_chat_completions(
        self,
        body: Dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Tuple[int, Dict[str, Any]]: ...

    async def close(self) -> None: ...


def _as_mapping(x: Any) -> Mapping[str, Any] | None:
    return x if isinstance(x, Mapping) else None


def _is_event_stream_response(resp: httpx.Response) -> bool:
    ct = resp.headers.get("content-type") or ""
    return "text/event-stream" in ct.lower()


def _merge_stream_options(body: Dict[str, Any]) -> None:
    """
    Ensure `stream_options.include_usage=True` in-place.

    vLLM supports OpenAI-style `stream_options={"include_usage": true}`.
    """
    so = _as_mapping(body.get("stream_options"))
    if so is None:
        body["stream_options"] = {"include_usage": True}
        return

    # Preserve other options but force include_usage=True.
    merged = dict(so)
    merged["include_usage"] = True
    body["stream_options"] = merged


@dataclass(slots=True)
class _ChoiceAccum:
    role: str = "assistant"
    parts: list[str] = field(default_factory=list)
    finish_reason: str | None = None

    def add_delta(self, delta: Mapping[str, Any]) -> None:
        r = delta.get("role")
        if isinstance(r, str) and r.strip():
            self.role = r.strip()

        c = delta.get("content")
        if isinstance(c, str) and c:
            self.parts.append(c)

    def content(self) -> str:
        return "".join(self.parts)


@dataclass(slots=True)
class _StreamReconstruction:
    """
    Reconstruct a non-streaming Chat Completions response from streamed chunks.

    This is intentionally conservative and focuses on the basic content path
    (choices[].delta.content). It does not attempt full tool-call merging.
    """

    completion_id: str = ""
    model: str = ""
    created: int = 0
    usage: Mapping[str, Any] | None = None
    choices_by_index: dict[int, _ChoiceAccum] = field(default_factory=dict)

    def ingest_chunk(self, chunk: Mapping[str, Any]) -> None:
        if not self.completion_id:
            cid = chunk.get("id")
            if isinstance(cid, str):
                self.completion_id = cid

        if not self.model:
            m = chunk.get("model")
            if isinstance(m, str):
                self.model = m

        if self.created == 0:
            cr = chunk.get("created")
            try:
                self.created = int(cr) if cr is not None else 0
            except Exception:
                self.created = 0

        u = _as_mapping(chunk.get("usage"))
        if u is not None:
            self.usage = u

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return

        for c in choices:
            cm = _as_mapping(c)
            if cm is None:
                continue

            idx_raw = cm.get("index")
            try:
                idx = int(idx_raw) if idx_raw is not None else 0
            except Exception:
                idx = 0

            acc = self.choices_by_index.get(idx)
            if acc is None:
                acc = _ChoiceAccum()
                self.choices_by_index[idx] = acc

            delta = _as_mapping(cm.get("delta")) or {}
            acc.add_delta(delta)

            fr = cm.get("finish_reason")
            if isinstance(fr, str) or fr is None:
                # Keep last finish_reason observed.
                acc.finish_reason = fr

    def build_chat_completion(self) -> Dict[str, Any]:
        # Produce stable ordering.
        indices = sorted(self.choices_by_index.keys())
        choices_out: list[dict[str, Any]] = []

        for idx in indices:
            acc = self.choices_by_index[idx]
            choices_out.append(
                {
                    "index": int(idx),
                    "message": {
                        "role": str(acc.role or "assistant"),
                        "content": str(acc.content()),
                    },
                    "finish_reason": acc.finish_reason,
                }
            )

        payload: Dict[str, Any] = {
            "id": str(self.completion_id),
            "object": "chat.completion",
            "created": int(self.created),
            "model": str(self.model),
            "choices": choices_out,
        }
        if self.usage is not None:
            payload["usage"] = dict(self.usage)

        return payload


@dataclass(slots=True)
class HttpOpenAIBackend:
    """
    Basic backend that forwards requests to an OpenAI-compatible HTTP server.

    Notes
    -----
    - Expects `base_url_v1` like "http://host:port/v1".
    - Does not attempt streaming pass-through to the client.
    - Attaches configured backend Authorization header (if api_key is set).
    - Internally, it can force upstream streaming to measure a TTFT-like proxy.
    """

    backend_id: str
    base_url_v1: str
    api_key: str = ""
    _http: httpx.AsyncClient = field(init=False, repr=False)
    _default_headers: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient()
        self._default_headers = {}
        if self.api_key.strip():
            self._default_headers["authorization"] = f"Bearer {self.api_key.strip()}"

    def _merge_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        out = dict(self._default_headers)
        if headers:
            # Allow router to add non-auth headers (run id etc).
            out.update(dict(headers))
        return out

    async def list_models(self) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url_v1}/models"
        r = await self._http.get(url, headers=self._merge_headers(None), timeout=10.0)
        try:
            payload = dict(r.json())
        except Exception:
            payload = {"error": {"message": "Non-JSON response from backend /models"}}
        return r.status_code, payload

    async def forward_chat_completions(
        self,
        body: Dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Forward /v1/chat/completions.

        Implementation strategy
        -----------------------
        We force upstream streaming and consume the SSE stream to:
        - measure a TTFT-like proxy (first meaningful delta arrival),
        - reconstruct a standard non-streaming Chat Completions JSON response.

        The measured monotonic timestamp is attached under the private key
        `_immas_t_first_token_monotonic` for the router to convert into `obs_latency_ms`.
        """
        url = f"{self.base_url_v1}/chat/completions"

        # Force upstream streaming so we can measure first chunk arrival time.
        forced_body: Dict[str, Any] = dict(body)
        forced_body["stream"] = True
        _merge_stream_options(forced_body)

        merged_headers = self._merge_headers(headers)

        try:
            async with self._http.stream(
                "POST", url, json=forced_body, headers=merged_headers, timeout=None
            ) as r:
                # If backend didn't actually stream, fall back to JSON parsing.
                if not _is_event_stream_response(r):
                    try:
                        payload = dict(await r.json())
                    except Exception:
                        payload = {
                            "error": {
                                "message": "Non-JSON response from backend /chat/completions"
                            }
                        }
                    return r.status_code, payload

                recon = _StreamReconstruction()
                t_first_event: float | None = None
                t_first_content: float | None = None

                async for line in r.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "data: <json>" or "data: [DONE]"
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break

                    if t_first_event is None:
                        t_first_event = float(time.monotonic())

                    try:
                        chunk_any = json.loads(data)
                    except Exception:
                        continue

                    chunk = _as_mapping(chunk_any)
                    if chunk is None:
                        continue

                    # If backend streams an error object, return it directly.
                    if "error" in chunk and isinstance(chunk.get("error"), Mapping):
                        payload_err = dict(chunk)
                        if t_first_event is not None:
                            payload_err[_IMMAS_T_FIRST_TOKEN_MONOTONIC_KEY] = float(
                                t_first_event
                            )
                        return r.status_code, payload_err

                    # Detect first "meaningful" content delta.
                    if t_first_content is None:
                        choices = chunk.get("choices")
                        if isinstance(choices, list):
                            for c in choices:
                                cm = _as_mapping(c)
                                if cm is None:
                                    continue
                                delta = _as_mapping(cm.get("delta"))
                                if delta is None:
                                    continue
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    t_first_content = float(time.monotonic())
                                    break

                    recon.ingest_chunk(chunk)

                payload = recon.build_chat_completion()

                # Prefer first content token; fall back to first event.
                t_first = (
                    t_first_content if t_first_content is not None else t_first_event
                )
                if t_first is not None:
                    payload[_IMMAS_T_FIRST_TOKEN_MONOTONIC_KEY] = float(t_first)

                return r.status_code, payload
        except Exception:
            # Preserve old behavior on unexpected streaming failures.
            r2 = await self._http.post(
                url, json=body, headers=self._merge_headers(headers), timeout=None
            )
            try:
                payload2 = dict(r2.json())
            except Exception:
                payload2 = {
                    "error": {
                        "message": "Non-JSON response from backend /chat/completions"
                    }
                }
            return r2.status_code, payload2

    async def close(self) -> None:
        await self._http.aclose()
