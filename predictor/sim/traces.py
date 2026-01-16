"""
In-memory trace store for fake chat completions.

We store debug traces keyed by the returned chat completion id so the client can
retrieve simulated TTFT/queue/stall/load values without changing the OpenAI
response schema (keeps compatibility with official OpenAI clients).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class ChatCompletionTrace:
    """Debug trace for a single chat completion."""

    completion_id: str
    created: int
    model: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Load snapshot at entry to the model handler (per-process)
    load_inflight: int
    load_rps: float

    # Simulator outputs
    sim_ttft_s: float
    sim_decode_s: float
    sim_queue_s: float
    sim_stall_s: float
    sim_warmup_s: float
    sim_total_s: float

    # Simulator load-derived stats
    effective_inflight: int
    utilization: float
    sim_rps: float

    # Actual server wall time spent inside the model handler (includes simulated sleep)
    server_wall_s: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "completion_id": self.completion_id,
            "created": self.created,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "load": {
                "inflight": self.load_inflight,
                "rps": self.load_rps,
            },
            "sim": {
                "ttft_s": self.sim_ttft_s,
                "decode_s": self.sim_decode_s,
                "queue_s": self.sim_queue_s,
                "stall_s": self.sim_stall_s,
                "warmup_s": self.sim_warmup_s,
                "total_s": self.sim_total_s,
            },
            "sim_load": {
                "effective_inflight": self.effective_inflight,
                "utilization": self.utilization,
                "rps": self.sim_rps,
            },
            "server_wall_s": self.server_wall_s,
        }


class InMemoryTraceStore:
    """
    Bounded in-memory trace store (FIFO eviction).

    Notes
    -----
    - This is per-process. With multiple Uvicorn workers, traces are not shared.
    - Intended for debugging / instrumentation, not durability.
    """

    def __init__(self, *, max_size: int = 50_000) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = int(max_size)
        self._lock = asyncio.Lock()
        self._data: "OrderedDict[str, ChatCompletionTrace]" = OrderedDict()

    async def put(self, trace: ChatCompletionTrace) -> None:
        """Insert a trace; evict oldest if over capacity."""
        async with self._lock:
            self._data[trace.completion_id] = trace
            self._data.move_to_end(trace.completion_id)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    async def get(self, completion_id: str) -> Optional[ChatCompletionTrace]:
        """Fetch a trace by id, or None if missing."""
        async with self._lock:
            return self._data.get(completion_id)
