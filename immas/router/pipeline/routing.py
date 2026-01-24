"""
immas.router.pipeline.routing

Routing logic for the router app.
"""

from __future__ import annotations

import asyncio

from immas.router.components.backend import HttpOpenAIBackend
from immas.router.state import RouterState


async def select_backends_round_robin(
    state: RouterState, n: int
) -> list[HttpOpenAIBackend]:
    """
    Select N backends in a single round-robin critical section.

    This is cheaper than acquiring the RR lock per request and also ensures
    deterministic batch ordering.
    """

    backends: list[HttpOpenAIBackend] = state.backends
    rr_lock: asyncio.Lock = state.rr_lock

    if n <= 0:
        return []
    if not backends:
        return []

    async with rr_lock:
        start_idx: int = int(state.rr_index)
        chosen = [backends[(start_idx + i) % len(backends)] for i in range(n)]
        state.rr_index = (start_idx + n) % len(backends)

    return chosen
