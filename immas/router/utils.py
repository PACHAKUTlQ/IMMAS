"""
immas.router.utils

General utilities for the router app.
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any, Mapping, cast

from fastapi import Request

from immas.openai.usage import ParsedUsage
from immas.router.types import ChatCompletionResult, PendingChatCompletion


_log = logging.getLogger(__name__)

# Conservative eviction heuristic thresholds.
_EVICT_KVMATCH_MIN = 0.8
_EVICT_OBS_CACHE_MAX = 0.10
_EVICT_MIN_PROMPT_TOKENS = 64


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _get_header(req: Request, name: str) -> str | None:
    return req.headers.get(name)


def _parse_turn_number(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _as_mapping(x: Any, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise TypeError(f"Expected mapping at {ctx}, got {type(x)!r}")

    return cast(Mapping[str, Any], x)


def _as_list(x: Any, *, ctx: str) -> list[Any]:
    if not isinstance(x, list):
        raise TypeError(f"Expected list at {ctx}, got {type(x)!r}")

    return x


def _as_str(x: Any, *, ctx: str) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        raise TypeError(f"Expected string at {ctx}, got {type(x)!r}")

    return x.strip()


def _as_bool(x: Any, *, ctx: str) -> bool:
    if isinstance(x, bool):
        return x

    raise TypeError(f"Expected bool at {ctx}, got {type(x)!r}")


def _as_int(x: Any, *, ctx: str) -> int:
    if isinstance(x, bool):
        # bool is a subclass of int; reject it explicitly.
        raise TypeError(f"Expected int at {ctx}, got bool")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        raise TypeError(f"Expected int at {ctx}, got non-integer float {x}")
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0
        try:
            return int(s)
        except Exception as e:
            raise TypeError(f"Expected int at {ctx}, got {x!r}") from e

    raise TypeError(f"Expected int at {ctx}, got {type(x)!r}")


def _as_float(x: Any, *, ctx: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"Expected float at {ctx}, got bool")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except Exception as e:
            raise TypeError(f"Expected float at {ctx}, got {x!r}") from e

    raise TypeError(f"Expected float at {ctx}, got {type(x)!r}")


def should_evict_router_prefix_cache(
    *,
    usage: ParsedUsage,
    turn_number: int,
    kvmatch_text: float,
    obs_cache_ratio: float,
) -> bool:
    """
    Decide whether to evict the router-side prefix cache record for this key.

    We only attempt eviction detection when the backend explicitly reports cached
    token accounting (`usage.cached_tokens_known == True`). Otherwise, cached_tokens=0
    could mean "unreported", and eviction detection would be wrong.

    The heuristic is intentionally conservative: we require near-perfect prefix match,
    sufficiently large prompts, and near-zero observed cache ratio.

    Returns
    -------
    bool
        True if the router should evict its prefix cache record and skip updating it.
    """

    if turn_number <= 1:
        # Eviction only matters when we expected reuse.
        return False

    if not usage.cached_tokens_known:
        return False

    if usage.prompt_tokens < _EVICT_MIN_PROMPT_TOKENS:
        return False

    if float(kvmatch_text) < _EVICT_KVMATCH_MIN:
        return False

    if float(obs_cache_ratio) > _EVICT_OBS_CACHE_MAX:
        return False

    return True


def try_set_future_result(
    fut: asyncio.Future[ChatCompletionResult], value: ChatCompletionResult
) -> None:
    """
    Best-effort set_result that never raises.

    This protects against races with cancellation / double completion.
    """

    if fut.done():
        return
    try:
        fut.set_result(value)
    except asyncio.InvalidStateError:
        return


def fail_pending_batch(
    batch: list[PendingChatCompletion],
    *,
    status_code: int,
    message: str,
) -> None:
    """
    Resolve all pending futures in a batch with an error response.

    This is used to enforce the router invariant that once a request is accepted
    into the micro-batcher, it will eventually be completed.

    Notes
    -----
    - Best-effort: ignores already-completed/cancelled futures.
    - Synchronous (no awaits): safe to call from exception handlers.
    """

    payload: dict[str, Any] = {"error": {"message": str(message)}}

    for p in batch:
        try_set_future_result(p.future, (int(status_code), dict(payload)))
