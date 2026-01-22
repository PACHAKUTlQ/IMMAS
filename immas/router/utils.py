"""
immas.router.utils

Utility functions for router.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

from fastapi import Request


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _get_header(req: Request, name: str) -> str:
    return (req.headers.get(name) or "").strip()


def _parse_turn_number(raw: str) -> int:
    try:
        n = int(raw)

        return n if n >= 0 else 0
    except Exception:
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
