"""
Structured JSONL logging for parallel experiments.

Writes one JSON object per request so logs remain parseable under concurrency.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True, slots=True)
class RequestLogRecord:
    """One request/response record suitable for JSONL."""

    run_id: str
    t_start_monotonic: float
    t_end_monotonic: float

    model: str
    source: str
    dialogue_id: str
    turn_number: int

    prompt_chars: int
    kvmatch: float
    client_inflight: int
    client_rps_1s: float

    pred_latency_ms: float
    pred_cost_tokens: float
    pred_perf_prob: float

    completion_id: str
    obs_latency_ms: float
    obs_total_tokens: int
    correct: bool

    # Optional server debug trace (flattened)
    srv_sim_ttft_s: Optional[float] = None
    srv_sim_total_s: Optional[float] = None
    srv_sim_stall_s: Optional[float] = None
    srv_utilization: Optional[float] = None
    srv_effective_inflight: Optional[int] = None
    srv_rps: Optional[float] = None

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


class AsyncJsonlLogger:
    """
    Async JSONL logger with a background writer task.

    This avoids interleaved prints and keeps disk writes serialized.
    """

    def __init__(
        self, path: str, *, append: bool = False, flush_every: int = 1
    ) -> None:
        self._path = Path(path)
        self._append = bool(append)
        self._flush_every = int(flush_every)
        if self._flush_every <= 0:
            raise ValueError("flush_every must be >= 1")

        self._q: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    async def __aenter__(self) -> "AsyncJsonlLogger":
        self._task = asyncio.create_task(self._writer_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def log(self, record: Union[RequestLogRecord, Dict[str, Any]]) -> None:
        """Enqueue one record (non-blocking except for queue backpressure)."""
        payload = (
            record.to_dict() if isinstance(record, RequestLogRecord) else dict(record)
        )
        await self._q.put(payload)

    async def close(self) -> None:
        """Flush and stop writer task."""
        if self._task is None:
            return
        await self._q.put(None)
        await self._task
        self._task = None

    async def _writer_loop(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        n = 0

        # Synchronous file IO in a single background task is usually fine for JSONL.
        with self._path.open(mode, encoding="utf-8") as f:
            while True:
                item = await self._q.get()
                if item is None:
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
                if n % self._flush_every == 0:
                    f.flush()
            f.flush()
