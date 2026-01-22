"""
immas.router.lifecycle

FastAPI application lifecycle management for the router.
"""

from __future__ import annotations

import asyncio
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI

from immas.common.load import AsyncLoadTracker
from immas.router.backend import HttpOpenAIBackend
from immas.router.batching import MicroBatchInfo, MicroBatcher
from immas.router.config import RouterAppConfig
from immas.router.logger import AsyncJsonlLogger
from immas.router.predictor import AsyncBackendPredictorPool
from immas.router.prefix_cache import TextPrefixCache
from immas.router.processing import handle_chat_batch
from immas.router.types import PendingChatCompletion
from immas.router.utils import fail_pending_batch


_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI, cfg: RouterAppConfig):
    """
    Application lifespan context manager for the router.

    Handles startup and shutdown of resources.
    """

    # Router feature cache (text prefix) - protect with lock.
    app.state.prefix_cache = TextPrefixCache()
    app.state.prefix_cache_lock = asyncio.Lock()

    # Router-side load tracker (inflight + RPS).
    app.state.load_tracker = AsyncLoadTracker(window_s=1.0)

    # Backends registry (ordered list for round-robin).
    app.state.backends = [
        HttpOpenAIBackend(
            backend_id=b.backend_id,
            base_url_v1=b.base_url_v1,
            api_key=b.api_key,
        )
        for b in cfg.backends
    ]
    app.state.backend_model_by_id = {b.backend_id: b.model for b in cfg.backends}

    # Independent predictor per backend.
    app.state.predictors = AsyncBackendPredictorPool(
        backend_ids=[b.backend_id for b in cfg.backends]
    )

    # Round-robin state.
    app.state.rr_lock = asyncio.Lock()
    app.state.rr_index = 0

    # Routing policy string (future: auction, etc.)
    app.state.routing_policy = cfg.router.routing

    # JSONL logger
    app.state.logger = AsyncJsonlLogger(
        cfg.router.log_path, append=cfg.router.log_append, flush_every=1
    )
    await app.state.logger.__aenter__()

    # Track inflight request tasks (so shutdown can cancel/await them).
    app.state.inflight_request_tasks = set()

    # Micro-batcher
    batching_cfg = cfg.router.batching
    if not batching_cfg.enabled:
        raise RuntimeError(
            "router.batching.enabled is false, but this router version requires batching. "
            "Set router.batching.enabled: true (or remove the key to use defaults)."
        )

    async def _batch_handler_safe(
        batch: list[PendingChatCompletion], info: MicroBatchInfo
    ) -> None:
        """
        MicroBatcher-facing wrapper that enforces completion of all batch items.

        This wrapper must never raise (except for cancellation), otherwise the
        MicroBatcher would drop the batch and callers would hang indefinitely.
        """

        try:
            await handle_chat_batch(app, batch, info)
        except asyncio.CancelledError:
            # Let cancellation propagate (shutdown), but do not mask it.
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

    app.state.chat_batcher = MicroBatcher[PendingChatCompletion](
        max_batch_size=int(batching_cfg.max_batch_size),
        max_wait_ms=float(batching_cfg.max_wait_ms),
        max_queue_size=int(batching_cfg.max_queue_size),
        handler=_batch_handler_safe,
        name="immas.chat_completions.microbatcher",
    )
    await app.state.chat_batcher.start()

    yield

    # Shutdown order matters:
    # 1) stop accepting new work (batcher)
    # 2) cancel/await inflight request tasks
    # 3) close logger
    # 4) close backends

    batcher: MicroBatcher[PendingChatCompletion] = app.state.chat_batcher
    await batcher.close()

    inflight: set[asyncio.Task[None]] = app.state.inflight_request_tasks
    if inflight:
        for t in list(inflight):
            t.cancel()
        await asyncio.gather(*list(inflight), return_exceptions=True)
        inflight.clear()

    await app.state.logger.close()
    for b in app.state.backends:
        await b.close()
