"""
immas.router.routing

Routing logic for the router app.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

from immas.router.backend import HttpOpenAIBackend


async def select_backends_round_robin(app: FastAPI, n: int) -> list[HttpOpenAIBackend]:
    """
    Select N backends in a single round-robin critical section.

    This is cheaper than acquiring the RR lock per request and also ensures
    deterministic batch ordering.
    """

    backends: list[HttpOpenAIBackend] = app.state.backends
    rr_lock: asyncio.Lock = app.state.rr_lock

    if n <= 0:
        return []
    if not backends:
        return []

    async with rr_lock:
        start_idx: int = int(app.state.rr_index)
        chosen = [backends[(start_idx + i) % len(backends)] for i in range(n)]
        app.state.rr_index = (start_idx + n) % len(backends)

    return chosen
