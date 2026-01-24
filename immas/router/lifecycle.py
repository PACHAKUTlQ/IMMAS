"""
immas.router.lifecycle

FastAPI application lifecycle management for the router.
"""

from __future__ import annotations

import asyncio
import logging

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from immas.common.load import AsyncLoadTracker
from immas.router.components.backend import HttpOpenAIBackend
from immas.router.components.batching import MicroBatchInfo, MicroBatcher
from immas.router.components.logger import AsyncJsonlLogger
from immas.router.components.predictor import AsyncBackendPredictorPool
from immas.router.components.prefix_cache import TextPrefixCache
from immas.router.pipeline.processing import handle_chat_batch
from immas.router.state import RouterState
from immas.router.types import PendingChatCompletion
from immas.router.utils import fail_pending_batch
from immas.router.warmup import warmup_router

if TYPE_CHECKING:
    from immas.router.config import RouterAppConfig


_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI, cfg: RouterAppConfig):
    """
    Application lifespan context manager for the router.

    Handles startup and shutdown of resources.
    """

    prefix_cache = TextPrefixCache()
    prefix_cache_lock = asyncio.Lock()

    # Router-global load tracker.
    load_tracker = AsyncLoadTracker(window_s=1.0)

    backends = [
        HttpOpenAIBackend(
            backend_id=b.backend_id,
            base_url_v1=b.base_url_v1,
            api_key=b.api_key,
        )
        for b in cfg.backends
    ]

    backend_model_by_id = {b.backend_id: b.model for b in cfg.backends}
    backend_capacity_by_id = {b.backend_id: int(b.capacity) for b in cfg.backends}

    # Per-backend concurrency control + load tracking.
    backend_semaphores: dict[str, asyncio.Semaphore] = {
        b.backend_id: asyncio.Semaphore(max(1, int(b.capacity))) for b in cfg.backends
    }
    backend_load_trackers: dict[str, AsyncLoadTracker] = {
        b.backend_id: AsyncLoadTracker(window_s=1.0) for b in cfg.backends
    }

    predictors = AsyncBackendPredictorPool(
        backend_ids=[b.backend_id for b in cfg.backends]
    )

    rr_lock = asyncio.Lock()
    rr_index = 0
    routing_policy = cfg.router.routing

    logger = AsyncJsonlLogger(
        cfg.router.log_path, append=cfg.router.log_append, flush_every=1
    )
    await logger.__aenter__()

    inflight_request_tasks: set[asyncio.Task[None]] = set()

    # Micro-batcher
    batching_cfg = cfg.router.batching
    if not batching_cfg.enabled:
        raise RuntimeError(
            "router.batching.enabled is false, but this router version requires batching."
        )

    # Use a local reference that is populated before the batcher starts.
    # This avoids any fragile dependency on app.state initialization order.
    router_state_ref: dict[str, RouterState] = {}

    async def _batch_handler_safe(
        batch: list[PendingChatCompletion], info: MicroBatchInfo
    ) -> None:
        """
        MicroBatcher-facing wrapper that enforces completion of all batch items.

        This wrapper must never raise (except for cancellation), otherwise the
        MicroBatcher would drop the batch and callers would hang indefinitely.
        """

        st = router_state_ref.get("state")
        if st is None:
            # Extremely defensive: should not happen because we start the batcher
            # only after setting router_state_ref["state"].
            fail_pending_batch(
                batch,
                status_code=503,
                message="Router not ready",
            )
            return

        try:
            await handle_chat_batch(st, batch, info)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.exception(
                "Batch handler crashed; failing all requests in batch_id=%s size=%s",
                info.batch_id,
                info.batch_size,
            )
            fail_pending_batch(
                batch,
                status_code=500,
                message=f"Internal router batch error: {type(e).__name__}",
            )
            # Swallow exception so MicroBatcher does not "drop" silently.

    chat_batcher = MicroBatcher[PendingChatCompletion](
        max_batch_size=int(batching_cfg.max_batch_size),
        max_wait_ms=float(batching_cfg.max_wait_ms),
        max_queue_size=int(batching_cfg.max_queue_size),
        handler=_batch_handler_safe,
        name="immas.chat_completions.microbatcher",
    )

    state = RouterState(
        cfg=cfg,
        backends=backends,
        backend_model_by_id=backend_model_by_id,
        backend_capacity_by_id=backend_capacity_by_id,
        backend_semaphores=backend_semaphores,
        backend_load_trackers=backend_load_trackers,
        predictors=predictors,
        prefix_cache=prefix_cache,
        prefix_cache_lock=prefix_cache_lock,
        chat_batcher=chat_batcher,
        logger=logger,
        inflight_request_tasks=inflight_request_tasks,
        load_tracker=load_tracker,
        routing_policy=routing_policy,
        rr_lock=rr_lock,
        rr_index=rr_index,
    )

    # Publish state before starting background workers.
    router_state_ref["state"] = state
    app.state.router_state = state

    # Warmup backends + bootstrap predictors (no JSONL logging).
    try:
        await warmup_router(state)
    except Exception:
        # Warmup should never prevent the router from starting.
        _log.exception("Warmup failed unexpectedly; continuing startup")

    # Start batcher only after warmup and state is ready.
    await chat_batcher.start()

    yield

    router_state: RouterState = app.state.router_state
    await router_state.chat_batcher.close()

    if router_state.inflight_request_tasks:
        for t in list(router_state.inflight_request_tasks):
            t.cancel()
        await asyncio.gather(
            *list(router_state.inflight_request_tasks), return_exceptions=True
        )
        router_state.inflight_request_tasks.clear()

    await router_state.logger.close()
    for b in router_state.backends:
        await b.close()
