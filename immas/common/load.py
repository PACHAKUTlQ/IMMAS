"""
immas.common.load

Async load tracking used by router (and optionally client/sim).
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
    """A consistent load snapshot."""

    inflight_requests: int
    rps: float
    window_s: float
    t_monotonic: float


class AsyncLoadTracker:
    """Tracks inflight requests and approximate RPS over a sliding window."""

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

    @asynccontextmanager
    async def track(self) -> AsyncIterator[LoadSnapshot]:
        async with self._lock:
            now = time.monotonic()
            self._inflight += 1
            self._starts.append(now)
            snap = self._snapshot_locked(now)

        try:
            yield snap
        finally:
            async with self._lock:
                self._inflight = max(0, self._inflight - 1)
