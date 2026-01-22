"""
immas.router.batching

Generic micro-batching infrastructure.

Motivation
----------
For real-time routing, we often want to "freeze" a short window of concurrent
requests into a micro-batch so a batch-level policy (e.g., auction / assignment)
can make decisions using the whole batch.

This module provides `MicroBatcher`, which:
- accepts items via an asyncio queue,
- collects up to `max_batch_size` items,
- waits up to `max_wait_ms` after the first item is received,
- calls an async handler with the collected batch.

The handler is expected to be fast and should typically schedule work onto other
tasks (so the batcher itself does not block on slow operations).
"""

from __future__ import annotations

import asyncio
import logging
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")
BatchHandler = Callable[[list[T], "MicroBatchInfo"], Awaitable[None]]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MicroBatchInfo:
    """Metadata for one emitted micro-batch."""

    batch_id: int
    batch_size: int
    t_batch_start_monotonic: float


class MicroBatcher(Generic[T]):
    """
    Micro-batcher with max-size and max-wait constraints.

    Parameters
    ----------
    max_batch_size
        Upper bound on number of items per batch. Must be >= 1.
    max_wait_ms
        Upper bound (milliseconds) on how long to wait after receiving the first
        item in a batch before emitting the batch. Must be >= 0.
    max_queue_size
        asyncio.Queue max size (0 means unbounded). Must be >= 0.
    handler
        Async function invoked for each produced batch. The handler should not
        block on long operations; it should schedule work and return quickly.
    name
        Optional name used for the background task (useful for debugging).
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_size: int,
        handler: BatchHandler[T],
        name: str = "microbatcher",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_wait_ms < 0:
            raise ValueError(f"max_wait_ms must be >= 0, got {max_wait_ms}")
        if max_queue_size < 0:
            raise ValueError(f"max_queue_size must be >= 0, got {max_queue_size}")

        self._max_batch_size = int(max_batch_size)
        self._max_wait_s = float(max_wait_ms) / 1000.0
        self._handler: BatchHandler[T] = handler
        self._name = str(name).strip() or "microbatcher"

        # We use Optional[T] so that None is a stop sentinel (items must be non-None).
        self._q: "asyncio.Queue[Optional[T]]" = asyncio.Queue(
            maxsize=int(max_queue_size)
        )

        self._closed: bool = False
        self._task: Optional[asyncio.Task[None]] = None
        self._batch_id: int = 0

    @property
    def closed(self) -> bool:
        """Whether the batcher is closed (no longer accepting items)."""
        return self._closed

    def qsize(self) -> int:
        """Current queue size (approximate)."""
        return int(self._q.qsize())

    async def start(self) -> None:
        """Start the background batching task (idempotent)."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name=self._name)

    def try_submit(self, item: T) -> bool:
        """
        Try to enqueue an item without blocking.

        Returns
        -------
        bool
            True if accepted, False if the batcher is closed or queue is full.
        """
        if self._closed:
            return False
        if item is None:
            raise ValueError(
                "MicroBatcher does not accept None items (None is reserved)"
            )

        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            return False
        return True

    async def submit(self, item: T) -> None:
        """
        Enqueue an item, waiting for queue space if needed.

        Notes
        -----
        For router request paths, prefer `try_submit` to avoid adding latency by
        blocking on a full queue. `submit` is provided for completeness/tests.
        """
        if self._closed:
            raise RuntimeError("MicroBatcher is closed")
        if item is None:
            raise ValueError(
                "MicroBatcher does not accept None items (None is reserved)"
            )
        await self._q.put(item)

    async def close(self) -> None:
        """Stop the batching loop and wait for it to exit (idempotent)."""
        if self._closed:
            # If already closed, still wait for task if present.
            if self._task is not None:
                await self._task
                self._task = None
            return

        self._closed = True

        if self._task is None:
            return

        # Signal stop. This may block if the queue is full; that's acceptable at shutdown.
        await self._q.put(None)

        try:
            await self._task
        finally:
            self._task = None

    async def _run_loop(self) -> None:
        try:
            while True:
                first = await self._q.get()
                if first is None:
                    break

                batch: list[T] = [first]
                t_start = time.perf_counter()
                deadline = t_start + self._max_wait_s

                while len(batch) < self._max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._q.get(), timeout=remaining)
                    except TimeoutError:
                        break

                    if item is None:
                        # Stop sentinel; process what we have then exit.
                        self._closed = True
                        break
                    batch.append(item)

                info = MicroBatchInfo(
                    batch_id=int(self._batch_id),
                    batch_size=int(len(batch)),
                    t_batch_start_monotonic=float(t_start),
                )
                self._batch_id += 1

                try:
                    await self._handler(batch, info)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.exception("MicroBatcher handler failed; dropping batch")
                if self._closed:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("MicroBatcher loop crashed")
