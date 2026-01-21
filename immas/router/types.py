"""
immas.router.types

Dataclasses for the router application.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from immas.router.components.backend import HttpOpenAIBackend


ChatCompletionResult = tuple[int, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PendingChatCompletion:
    """One pending /v1/chat/completions request waiting to be processed."""

    run_id: str
    dialogue_id: str
    turn_number: int
    source: str
    body: dict[str, Any]
    t_enqueued_monotonic: float
    future: asyncio.Future[ChatCompletionResult]


@dataclass(frozen=True, slots=True)
class PreparedChatCompletion:
    """
    One queued request with batch metadata and pre-fetched router cache state.

    `cached_text_by_backend` is read under the router cache lock in the batcher,
    so per-request processing does not need to lock for cache *reads*.
    """

    pending: PendingChatCompletion
    assigned_backend: HttpOpenAIBackend
    cached_text_by_backend: dict[str, str | None]
    batch_id: int
    batch_size: int
