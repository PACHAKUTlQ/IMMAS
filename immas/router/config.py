from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

from immas.router.utils import (
    _as_mapping,
    _as_list,
    _as_str,
    _as_bool,
    _as_int,
    _as_float,
)

RoutingPolicy = Literal["round_robin"]


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """One backend entry in router config."""

    backend_id: str
    base_url_v1: str
    api_key: str
    model: str


@dataclass(frozen=True, slots=True)
class RouterBatchingConfig:
    """
    Micro-batching configuration.

    Notes
    -----
    - max_wait_ms bounds the waiting time for the first request in a batch.
    - max_batch_size bounds how many requests can be included in one batch.
    - max_queue_size provides backpressure (0 means unbounded).
    """

    enabled: bool = True
    max_batch_size: int = 16
    max_wait_ms: float = 10.0
    max_queue_size: int = 4096


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Router settings."""

    log_path: str = "router_run.jsonl"
    log_append: bool = False
    routing: RoutingPolicy = "round_robin"
    batching: RouterBatchingConfig = field(default_factory=RouterBatchingConfig)


@dataclass(frozen=True, slots=True)
class RouterAppConfig:
    """Full router application configuration."""

    router: RouterConfig
    backends: list[BackendConfig]


def load_router_app_config(path: str) -> RouterAppConfig:
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

    batching_raw = _as_mapping(
        router_raw.get("batching", {}), ctx="root.router.batching"
    )
    batching_enabled = _as_bool(
        batching_raw.get("enabled", True), ctx="router.batching.enabled"
    )
    max_batch_size = _as_int(
        batching_raw.get("max_batch_size", 16), ctx="router.batching.max_batch_size"
    )
    max_wait_ms = _as_float(
        batching_raw.get("max_wait_ms", 10.0), ctx="router.batching.max_wait_ms"
    )
    max_queue_size = _as_int(
        batching_raw.get("max_queue_size", 4096), ctx="router.batching.max_queue_size"
    )

    if max_batch_size < 1:
        raise ValueError(
            f"router.batching.max_batch_size must be >= 1, got {max_batch_size}"
        )
    if max_wait_ms < 0:
        raise ValueError(f"router.batching.max_wait_ms must be >= 0, got {max_wait_ms}")
    if max_queue_size < 0:
        raise ValueError(
            f"router.batching.max_queue_size must be >= 0, got {max_queue_size}"
        )

    backends_raw = _as_list(root.get("backends"), ctx="root.backends")
    backends: list[BackendConfig] = []
    seen_ids: set[str] = set()

    for i, b in enumerate(backends_raw):
        bm = _as_mapping(b, ctx=f"backends[{i}]")
        backend_id = _as_str(bm.get("id"), ctx=f"backends[{i}].id")
        base_url_v1 = _as_str(bm.get("base_url_v1"), ctx=f"backends[{i}].base_url_v1")
        api_key = _as_str(bm.get("api_key", ""), ctx=f"backends[{i}].api_key")
        model = _as_str(bm.get("model"), ctx=f"backends[{i}].model")

        if not backend_id:
            raise ValueError(f"Missing/empty backends[{i}].id")
        if backend_id in seen_ids:
            raise ValueError(f"Duplicate backend id in config: {backend_id!r}")
        seen_ids.add(backend_id)

        if not base_url_v1:
            raise ValueError(f"Missing/empty backends[{i}].base_url_v1")
        base_url_v1 = base_url_v1.rstrip("/")

        if not model:
            raise ValueError(f"Missing/empty backends[{i}].model")

        backends.append(
            BackendConfig(
                backend_id=backend_id,
                base_url_v1=base_url_v1,
                api_key=api_key,
                model=model,
            )
        )

    if not backends:
        raise ValueError("Config must contain at least one backend in `backends:`")

    return RouterAppConfig(
        router=RouterConfig(
            log_path=log_path,
            log_append=log_append,
            routing=routing,
            batching=RouterBatchingConfig(
                enabled=bool(batching_enabled),
                max_batch_size=int(max_batch_size),
                max_wait_ms=float(max_wait_ms),
                max_queue_size=int(max_queue_size),
            ),
        ),
        backends=backends,
    )
