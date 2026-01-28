"""
immas.router.state

Dataclass for router application state.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from immas.router.components.performance import PerformanceEvaluator
from immas.router.pricing import BackendTokenPrices

if TYPE_CHECKING:
    from immas.common.load import AsyncLoadTracker
    from immas.router.components.backend import HttpOpenAIBackend
    from immas.router.components.batching import MicroBatcher
    from immas.router.components.detailed_csv import AsyncDetailedCsvLogger
    from immas.router.components.logger import AsyncJsonlLogger
    from immas.router.components.predictor import AsyncBackendPredictorPool
    from immas.router.components.prefix_cache import TextPrefixCache
    from immas.router.config import RouterAppConfig
    from immas.router.types import PendingChatCompletion


@dataclass(slots=True)
class RouterState:
    """A container for all router-specific state."""

    cfg: "RouterAppConfig"
    backends: list["HttpOpenAIBackend"]
    backend_model_by_id: dict[str, str]

    # Capacity and per-backend runtime controls.
    backend_capacity_by_id: dict[str, int]
    backend_semaphores: dict[str, asyncio.Semaphore]
    backend_load_trackers: dict[str, "AsyncLoadTracker"]

    # Backend token pricing used for observed cost computation.
    backend_prices_by_id: dict[str, BackendTokenPrices]

    predictors: "AsyncBackendPredictorPool"
    perf_evaluator: PerformanceEvaluator
    detailed_csv_logger: Optional["AsyncDetailedCsvLogger"]

    prefix_cache: "TextPrefixCache"
    prefix_cache_lock: asyncio.Lock
    chat_batcher: "MicroBatcher[PendingChatCompletion]"
    logger: "AsyncJsonlLogger"
    inflight_request_tasks: set[asyncio.Task[None]]

    # Router-global load tracking.
    load_tracker: "AsyncLoadTracker"

    routing_policy: str
    rr_lock: asyncio.Lock
    rr_index: int
