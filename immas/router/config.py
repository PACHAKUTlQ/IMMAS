"""
immas.router.config

Router config file parser
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

from immas.router.utils import (
    _as_bool,
    _as_float,
    _as_int,
    _as_list,
    _as_mapping,
    _as_str,
)

RoutingPolicy = Literal["round_robin", "auction"]


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """One backend entry in router config."""

    backend_id: str
    base_url_v1: str
    api_key: str
    model: str

    # Capacity is used by the auction and by per-backend concurrency control.
    # If omitted, we default to a reasonably large value for backward compatibility.
    capacity: int = 128


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
class RouterWarmupConfig:
    """
    Startup warmup configuration.

    Warmup runs internally during router startup and does not affect client-visible
    behavior or router JSONL logs.
    """

    enabled: bool = True

    # Dataset selection
    coqa_split: str = "validation"
    max_dialogues: int = 1
    max_turns_per_dialogue: int = 2
    shuffle_dialogues: bool = False
    seed: int = 0

    # Request behavior
    max_concurrency: int = 4
    timeout_s: float = 30.0
    max_tokens: int = 16

    # A short marker prepended to the warmup system message. A per-startup nonce
    # is appended at runtime.
    system_prefix: str = "IMMAS_WARMUP"


@dataclass(frozen=True, slots=True)
class RouterAuctionConfig:
    """
    Auction configuration.

    Note: VCG is always computed for matched tasks; it is part of the mechanism.
    """

    # Welfare terms (aligned with the reference simulation's structure).
    quality_scale: float = 100.0
    latency_scale: float = 5.0
    cost_scale: float = 0.02

    # Default client preference δ in [0,1].
    delta_default: float = 0.5

    # First-pass edge pruning: only edges with welfare > min_welfare_edge are included.
    # Router will still ensure at least one edge per task, and will fallback safely.
    min_welfare_edge: float = 0.0

    # Float -> int scaling for MCMF edge costs.
    mcmf_scale: int = 1000

    # Optional congestion penalty term applied in routing layer:
    # w_{ij} -= congestion_penalty * inflight_j / max(1, capacity_j)
    congestion_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Router settings."""

    log_path: str = "router_run.jsonl"
    log_append: bool = False
    routing: RoutingPolicy = "round_robin"
    batching: RouterBatchingConfig = field(default_factory=RouterBatchingConfig)
    warmup: RouterWarmupConfig = field(default_factory=RouterWarmupConfig)
    auction: RouterAuctionConfig = field(default_factory=RouterAuctionConfig)


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
    if routing_raw not in ("round_robin", "auction"):
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

    warmup_raw = _as_mapping(router_raw.get("warmup", {}), ctx="root.router.warmup")
    warmup_enabled = _as_bool(
        warmup_raw.get("enabled", True), ctx="router.warmup.enabled"
    )
    coqa_split = _as_str(
        warmup_raw.get("coqa_split", "validation"), ctx="router.warmup.coqa_split"
    )
    warmup_max_dialogues = _as_int(
        warmup_raw.get("max_dialogues", 1), ctx="router.warmup.max_dialogues"
    )
    warmup_max_turns = _as_int(
        warmup_raw.get("max_turns_per_dialogue", 2),
        ctx="router.warmup.max_turns_per_dialogue",
    )
    warmup_shuffle = _as_bool(
        warmup_raw.get("shuffle_dialogues", False),
        ctx="router.warmup.shuffle_dialogues",
    )
    warmup_seed = _as_int(warmup_raw.get("seed", 0), ctx="router.warmup.seed")

    warmup_max_concurrency = _as_int(
        warmup_raw.get("max_concurrency", 4), ctx="router.warmup.max_concurrency"
    )
    warmup_timeout_s = _as_float(
        warmup_raw.get("timeout_s", 30.0), ctx="router.warmup.timeout_s"
    )
    warmup_max_tokens = _as_int(
        warmup_raw.get("max_tokens", 16), ctx="router.warmup.max_tokens"
    )
    system_prefix = _as_str(
        warmup_raw.get("system_prefix", "IMMAS_WARMUP"),
        ctx="router.warmup.system_prefix",
    )

    if warmup_max_dialogues < 0:
        raise ValueError(
            f"router.warmup.max_dialogues must be >= 0, got {warmup_max_dialogues}"
        )
    if warmup_max_turns < 0:
        raise ValueError(
            f"router.warmup.max_turns_per_dialogue must be >= 0, got {warmup_max_turns}"
        )
    if warmup_max_concurrency < 1:
        raise ValueError(
            f"router.warmup.max_concurrency must be >= 1, got {warmup_max_concurrency}"
        )
    if warmup_timeout_s <= 0:
        raise ValueError(f"router.warmup.timeout_s must be > 0, got {warmup_timeout_s}")
    if warmup_max_tokens < 1:
        raise ValueError(
            f"router.warmup.max_tokens must be >= 1, got {warmup_max_tokens}"
        )

    auction_raw = _as_mapping(router_raw.get("auction", {}), ctx="root.router.auction")
    quality_scale = _as_float(
        auction_raw.get("quality_scale", 100.0), ctx="router.auction.quality_scale"
    )
    latency_scale = _as_float(
        auction_raw.get("latency_scale", 5.0), ctx="router.auction.latency_scale"
    )
    cost_scale = _as_float(
        auction_raw.get("cost_scale", 0.02), ctx="router.auction.cost_scale"
    )
    delta_default = _as_float(
        auction_raw.get("delta_default", 0.5), ctx="router.auction.delta_default"
    )
    min_welfare_edge = _as_float(
        auction_raw.get("min_welfare_edge", 0.0), ctx="router.auction.min_welfare_edge"
    )
    mcmf_scale = _as_int(
        auction_raw.get("mcmf_scale", 1000), ctx="router.auction.mcmf_scale"
    )
    congestion_penalty = _as_float(
        auction_raw.get("congestion_penalty", 0.0),
        ctx="router.auction.congestion_penalty",
    )

    if mcmf_scale <= 0:
        raise ValueError(f"router.auction.mcmf_scale must be > 0, got {mcmf_scale}")
    if not (0.0 <= delta_default <= 1.0):
        raise ValueError(
            f"router.auction.delta_default must be in [0,1], got {delta_default}"
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
        capacity = _as_int(bm.get("capacity", 128), ctx=f"backends[{i}].capacity")

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

        if capacity < 1:
            raise ValueError(f"backends[{i}].capacity must be >= 1, got {capacity}")

        backends.append(
            BackendConfig(
                backend_id=backend_id,
                base_url_v1=base_url_v1,
                api_key=api_key,
                model=model,
                capacity=int(capacity),
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
            warmup=RouterWarmupConfig(
                enabled=bool(warmup_enabled),
                coqa_split=str(coqa_split or "validation"),
                max_dialogues=int(warmup_max_dialogues),
                max_turns_per_dialogue=int(warmup_max_turns),
                shuffle_dialogues=bool(warmup_shuffle),
                seed=int(warmup_seed),
                max_concurrency=int(warmup_max_concurrency),
                timeout_s=float(warmup_timeout_s),
                max_tokens=int(warmup_max_tokens),
                system_prefix=str(system_prefix or "IMMAS_WARMUP"),
            ),
            auction=RouterAuctionConfig(
                quality_scale=float(quality_scale),
                latency_scale=float(latency_scale),
                cost_scale=float(cost_scale),
                delta_default=float(delta_default),
                min_welfare_edge=float(min_welfare_edge),
                mcmf_scale=int(mcmf_scale),
                congestion_penalty=float(congestion_penalty),
            ),
        ),
        backends=backends,
    )


def load_cfg_from_env() -> RouterAppConfig:
    """
    Load router config from YAML path in env IMMAS_ROUTER_CONFIG.

    This is the primary configuration mechanism.
    """

    path = (os.environ.get("IMMAS_ROUTER_CONFIG") or "").strip()
    if not path:
        raise RuntimeError(
            "Missing IMMAS_ROUTER_CONFIG. Please set it to a YAML config file path."
        )

    return load_router_app_config(path)
