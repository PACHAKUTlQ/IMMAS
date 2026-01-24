"""
immas.router.types

Dataclasses for the router application.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from immas.router.components.backend import HttpOpenAIBackend
from immas.router.components.logger import RouterBackendScore
from immas.router.components.predictor import PredictorInput


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
    One request prepared at batch level for per-request processing.

    Batch-level responsibilities (already done before creating this object):
    - serialize prompt
    - read router prefix cache under one lock
    - compute per-backend prefix match proxy
    - compute per-backend predictor outputs
    - build `backend_scores` list for logging/analysis

    Per-request processing then:
    - forwards to assigned backend
    - updates predictor/cache on observed outcome
    - logs and resolves the future
    """

    pending: PendingChatCompletion
    assigned_backend: HttpOpenAIBackend

    # Effective model name that will be forwarded to the backend.
    effective_model: str

    # Deterministic prompt representation for caching/proxy features and analysis.
    prompt_repr: str
    prompt_chars: int

    # Precomputed per-backend scores (prefix proxy + predictor outputs).
    backend_scores: list[RouterBackendScore]

    # Predictor input for the chosen backend (for online update).
    chosen_predictor_input: PredictorInput

    # Micro-batching metadata
    batch_id: int
    batch_size: int
