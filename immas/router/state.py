"""
immas.router.state

Dataclass for router application state.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from immas.common.load import AsyncLoadTracker
    from immas.router.backend import HttpOpenAIBackend
    from immas.router.batching import MicroBatcher
    from immas.router.config import RouterAppConfig
    from immas.router.logger import AsyncJsonlLogger
    from immas.router.predictor import AsyncBackendPredictorPool
    from immas.router.prefix_cache import TextPrefixCache
    from immas.router.types import PendingChatCompletion


@dataclass(slots=True)
class RouterState:
    """A container for all router-specific state."""

    cfg: RouterAppConfig
    backends: list[HttpOpenAIBackend]
    backend_model_by_id: dict[str, str]
    predictors: AsyncBackendPredictorPool
    prefix_cache: TextPrefixCache
    prefix_cache_lock: asyncio.Lock
    chat_batcher: MicroBatcher[PendingChatCompletion]
    logger: AsyncJsonlLogger
    inflight_request_tasks: set[asyncio.Task[None]]
    load_tracker: AsyncLoadTracker
    routing_policy: str
    rr_lock: asyncio.Lock
    rr_index: int
