"""
immas.router.telemetry

Internal telemetry keys used to pass router-only measurements through backend payloads.

Important
---------
These fields are strictly internal to the router process and must never be returned
to external clients. Call sites should remove them (e.g. via pop) before replying.

We use this mechanism to support TTFT-like latency measurement while still returning
a non-streaming Chat Completions JSON response to clients.
"""

from __future__ import annotations

from typing import Any


TTFT_MONOTONIC_KEY: str = "_immas_t_first_token_monotonic"


def pop_ttft_monotonic(payload: dict[str, Any] | None) -> float | None:
    """
    Pop and return the internal TTFT monotonic timestamp from a response payload.

    Parameters
    ----------
    payload
        Mutable response payload dict, or None.

    Returns
    -------
    float | None
        Monotonic timestamp if present and parseable; otherwise None.
    """

    if payload is None:
        return None

    raw = payload.pop(TTFT_MONOTONIC_KEY, None)
    if raw is None:
        return None

    try:
        return float(raw)
    except Exception:
        return None


def set_ttft_monotonic(payload: dict[str, Any], t_monotonic: float) -> None:
    """
    Set the internal TTFT monotonic timestamp in a response payload.

    This is intended to be used by backend forwarders after reconstructing a
    non-streaming response from an upstream streaming response.
    """

    payload[TTFT_MONOTONIC_KEY] = float(t_monotonic)
