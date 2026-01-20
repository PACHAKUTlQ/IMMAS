"""
immas.router.config

YAML configuration loader for the router.

Design goals
------------
- Production-friendly: explicit validation, strict typing, clear errors.
- Supports multiple backends, each with its own OpenAI-compatible base URL and API key.
- Router does not authenticate clients; it uses backend API keys from this config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import yaml


RoutingPolicy = Literal["round_robin"]


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """One backend entry in router config."""

    backend_id: str
    base_url_v1: str
    api_key: str


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Router settings."""

    log_path: str = "router_run.jsonl"
    log_append: bool = False
    routing: RoutingPolicy = "round_robin"


@dataclass(frozen=True, slots=True)
class RouterAppConfig:
    """Full router application configuration."""

    router: RouterConfig
    backends: list[BackendConfig]


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


def load_router_app_config(path: str) -> RouterAppConfig:
    """
    Load router YAML configuration from `path`.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError / TypeError
        If config schema is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Router config not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _as_mapping(raw, ctx="root")

    router_raw = _as_mapping(root.get("router", {}), ctx="root.router")
    log_path = _as_str(
        router_raw.get("log_path", "router_run.jsonl"), ctx="router.log_path"
    )
    log_append = _as_bool(router_raw.get("log_append", False), ctx="router.log_append")

    routing_raw = (
        _as_str(router_raw.get("routing", "round_robin"), ctx="router.routing")
        or "round_robin"
    )
    if routing_raw not in ("round_robin",):
        raise ValueError(f"Unsupported routing policy: {routing_raw!r}")
    routing = cast(RoutingPolicy, routing_raw)

    backends_raw = _as_list(root.get("backends"), ctx="root.backends")
    backends: list[BackendConfig] = []
    seen_ids: set[str] = set()

    for i, b in enumerate(backends_raw):
        bm = _as_mapping(b, ctx=f"backends[{i}]")
        backend_id = _as_str(bm.get("id"), ctx=f"backends[{i}].id")
        base_url_v1 = _as_str(bm.get("base_url_v1"), ctx=f"backends[{i}].base_url_v1")
        api_key = _as_str(bm.get("api_key", ""), ctx=f"backends[{i}].api_key")

        if not backend_id:
            raise ValueError(f"Missing/empty backends[{i}].id")
        if backend_id in seen_ids:
            raise ValueError(f"Duplicate backend id in config: {backend_id!r}")
        seen_ids.add(backend_id)

        if not base_url_v1:
            raise ValueError(f"Missing/empty backends[{i}].base_url_v1")
        base_url_v1 = base_url_v1.rstrip("/")

        backends.append(
            BackendConfig(
                backend_id=backend_id,
                base_url_v1=base_url_v1,
                api_key=api_key,
            )
        )

    if not backends:
        raise ValueError("Config must contain at least one backend in `backends:`")

    return RouterAppConfig(
        router=RouterConfig(log_path=log_path, log_append=log_append, routing=routing),
        backends=backends,
    )
