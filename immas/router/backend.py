"""
immas.router.backend

Backend abstraction for OpenAI-compatible servers.
Uses a simple HTTP JSON forwarder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Tuple

import httpx


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


@dataclass(slots=True)
class HttpOpenAIBackend:
    """
    Basic backend that forwards requests to an OpenAI-compatible HTTP server.

    Notes
    -----
    - Expects `base_url_v1` like "http://host:port/v1".
    - Does not attempt streaming pass-through.
    """

    backend_id: str
    base_url_v1: str
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient()

    async def list_models(self) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url_v1}/models"
        r = await self._http.get(url, timeout=10.0)
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
        url = f"{self.base_url_v1}/chat/completions"
        r = await self._http.post(
            url, json=body, headers=dict(headers or {}), timeout=None
        )
        try:
            payload = dict(r.json())
        except Exception:
            payload = {
                "error": {"message": "Non-JSON response from backend /chat/completions"}
            }
        return r.status_code, payload

    async def close(self) -> None:
        await self._http.aclose()
