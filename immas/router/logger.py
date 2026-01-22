from __future__ import annotations

import asyncio
import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True, slots=True)
class RouterBackendScore:
    """
    Per-backend router-time score record.

    This is computed before routing to any single backend (i.e., it is suitable
    for later auction/batching logic where each request is scored against each backend).
    """

    backend_id: str
    model: str

    # Router-known prefix-cache proxy feature (backend-specific, because the router
    # tracks per-backend conversation history).
    cached_prompt_chars: int
    kvmatch_lcp_chars: int
    kvmatch_text: float

    # Predictor outputs (means only; std is currently unused/dummy in the model).
    pred_latency_ms: float
    pred_cost_tokens: float
    pred_perf_prob: float
    pred_cache_ratio: float


@dataclass(frozen=True, slots=True)
class RouterLogRecord:
    """One router request/response record suitable for JSONL."""

    run_id: str
    t_start_monotonic: float
    t_end_monotonic: float

    # Micro-batching metadata
    batch_id: int
    batch_size: int
    queue_wait_ms: float

    backend_id: str
    backend_base_url_v1: str

    model: str
    source: str
    dialogue_id: str
    turn_number: int

    # Router-known decision-time features (for the chosen backend)
    prompt_chars: int
    cached_prompt_chars: int
    kvmatch_lcp_chars: int
    kvmatch_text: float
    router_inflight: int
    router_rps_1s: float

    # Predictions (for the chosen backend)
    pred_latency_ms: float
    pred_cost_tokens: float
    pred_perf_prob: float
    pred_cache_ratio: float

    # Predictions for all backends (for future auction/batching).
    backend_scores: list[RouterBackendScore]

    # Observations
    completion_id: str
    obs_latency_ms: float

    obs_prompt_tokens: int
    obs_completion_tokens: int
    obs_total_tokens: int
    obs_cached_tokens: int
    obs_cache_ratio: float

    correct: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""

        return asdict(self)


class AsyncJsonlLogger:
    """Async JSONL logger with a single background writer task."""

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

    async def log(self, record: Union[RouterLogRecord, Dict[str, Any]]) -> None:
        payload = (
            record.to_dict() if isinstance(record, RouterLogRecord) else dict(record)
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
