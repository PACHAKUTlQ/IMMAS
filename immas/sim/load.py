"""
Async load tracking for the fake vLLM/OpenAI server.

We track:
- in-flight requests (per-process)
- approximate recent request rate over a sliding window (requests/second)

This is intentionally per-process. If run multiple Uvicorn workers,
each worker will track its own load.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Deque


@dataclass(frozen=True, slots=True)
class LoadSnapshot:
    """
    Snapshot of server load at a point in time.

    Attributes
    ----------
    inflight_requests
        Number of requests currently being processed by this process.
    rps
        Approximate request rate (requests/second) over the last `window_s`.
    window_s
        Sliding window length used to compute `rps`.
    t_monotonic
        Monotonic timestamp at which this snapshot was taken.
    """

    inflight_requests: int
    rps: float
    window_s: float
    t_monotonic: float


class AsyncLoadTracker:
    """
    Lightweight per-process load tracker for async web servers.

    Notes
    -----
    - Use `async with tracker.track():` around the work counted as "in-flight".
    - The tracker holds a lock only during counter updates, not during request work.
    """

    def __init__(self, *, window_s: float = 1.0) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self._window_s = float(window_s)
        self._lock = asyncio.Lock()
        self._inflight: int = 0
        self._starts: Deque[float] = deque()

    def _purge_old_locked(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._starts and self._starts[0] < cutoff:
            self._starts.popleft()

    def _snapshot_locked(self, now: float) -> LoadSnapshot:
        self._purge_old_locked(now)
        rps = float(len(self._starts)) / self._window_s
        return LoadSnapshot(
            inflight_requests=self._inflight,
            rps=rps,
            window_s=self._window_s,
            t_monotonic=now,
        )

    async def snapshot(self) -> LoadSnapshot:
        """Get a consistent snapshot of current load."""
        async with self._lock:
            now = time.monotonic()
            return self._snapshot_locked(now)

    @asynccontextmanager
    async def track(self) -> AsyncIterator[LoadSnapshot]:
        """
        Context manager that increments in-flight on entry and decrements on exit.

        Yields
        ------
        LoadSnapshot
            Snapshot taken immediately after incrementing in-flight and recording start time.
        """
        async with self._lock:
            now = time.monotonic()
            self._inflight += 1
            self._starts.append(now)
            snap = self._snapshot_locked(now)

        try:
            yield snap
        finally:
            async with self._lock:
                # Defensive clamp in case of unexpected cancellation interactions.
                self._inflight = max(0, self._inflight - 1)
