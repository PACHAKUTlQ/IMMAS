"""
immas.router.types

Dataclasses for the router application.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Any, Optional

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

    # Auction metadata (meaningful when routing policy is 'auction').
    routing_policy: str
    auction_matched: bool
    auction_total_welfare: float
    chosen_client_valuation: float
    chosen_base_cost: float
    chosen_welfare: float
    vcg_fee: Optional[float]
    vcg_total_payment: Optional[float]
