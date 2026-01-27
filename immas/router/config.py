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

RoutingPolicy = Literal["round_robin", "auction", "llmrouter"]
PerformanceEvaluatorKind = Literal["rouge", "token_span"]


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

    # Token pricing (arbitrary per-token units; only relative values matter).
    #
    # This enables differentiating:
    # - uncached prompt tokens (input_token_price),
    # - cached prompt tokens (cached_input_token_price),
    # - completion tokens (output_token_price).
    #
    # Defaults preserve the old behavior where "cost ~= total tokens" and does not
    # privilege cache hits.
    input_token_price: float = 1.0
    cached_input_token_price: float = 1.0
    output_token_price: float = 1.0


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
    max_tokens: int = 1000

    # A short marker prepended to the warmup system message. A per-startup nonce
    # is appended at runtime.
    system_prefix: str = "IMMAS_WARMUP"


@dataclass(frozen=True, slots=True)
class RouterPerformanceConfig:
    """
    Online performance evaluation configuration.

    When enabled, the router computes a real-time `correct` signal using the CoQA
    dataset gold answers.

    Evaluators
    ----------
    - rouge: dataset-backed evaluation using ROUGE between gold answer and the model's
      extracted last-line answer.
    - token_span: dataset-backed token-span substring match (fast, deterministic),
      on the same "last line" answer after numeric normalization.

    Notes
    -----
    When disabled, the router uses AlwaysCorrectEvaluator (placeholder) because in
    real non-dataset traffic the gold answer may be unknown at routing time.
    """

    enabled: bool = False

    # Which evaluator to use when enabled.
    evaluator: PerformanceEvaluatorKind = "rouge"

    # Dataset split to load for gold answers.
    coqa_split: str = "validation"

    # ROUGE metric and threshold for correctness (used when evaluator="rouge").
    rouge_metric: str = "rouge-l"  # "rouge-1" | "rouge-2" | "rouge-l"
    rouge_f1_threshold: float = 0.3
    lowercase: bool = True


@dataclass(frozen=True, slots=True)
class RouterDetailedCsvConfig:
    """
    Optional clean CSV logging of dataset-level details.

    This logger is intended for validating ROUGE behavior and inspecting:
    story/question/gold answer vs model answer with per-metric ROUGE F1 scores.

    Notes
    -----
    - This CSV is additive and independent from the JSONL router log.
    - Large fields (story/answers) are sanitized to keep one CSV record per line.
    """

    enabled: bool = False
    path: str = "router_detailed_answers.csv"
    append: bool = False
    flush_every: int = 1


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
class RouterLLMRouterConfig:
    """
    LLMRouter integration settings.
    """

    name: str = ""
    config_path: str = ""
    load_model_path: str = ""
    model_name_to_backend_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Router settings."""

    log_path: str = "router_run.jsonl"
    log_append: bool = False
    routing: RoutingPolicy = "round_robin"
    batching: RouterBatchingConfig = field(default_factory=RouterBatchingConfig)
    warmup: RouterWarmupConfig = field(default_factory=RouterWarmupConfig)
    performance: RouterPerformanceConfig = field(
        default_factory=RouterPerformanceConfig
    )
    detailed_csv: RouterDetailedCsvConfig = field(
        default_factory=RouterDetailedCsvConfig
    )
    auction: RouterAuctionConfig = field(default_factory=RouterAuctionConfig)
    llmrouter: RouterLLMRouterConfig = field(default_factory=RouterLLMRouterConfig)


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
    if routing_raw not in ("round_robin", "auction", "llmrouter"):
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
        warmup_raw.get("max_tokens", 1000), ctx="router.warmup.max_tokens"
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

    perf_raw = _as_mapping(
        router_raw.get("performance", {}), ctx="root.router.performance"
    )
    perf_enabled = _as_bool(
        perf_raw.get("enabled", False), ctx="router.performance.enabled"
    )
    perf_evaluator_raw = (
        _as_str(perf_raw.get("evaluator", "rouge"), ctx="router.performance.evaluator")
        or "rouge"
    ).lower()
    if perf_evaluator_raw not in ("rouge", "token_span"):
        raise ValueError(
            "router.performance.evaluator must be one of: rouge, token_span; "
            f"got {perf_evaluator_raw!r}"
        )
    perf_evaluator = cast(PerformanceEvaluatorKind, perf_evaluator_raw)

    perf_coqa_split = _as_str(
        perf_raw.get("coqa_split", "validation"), ctx="router.performance.coqa_split"
    )
    rouge_metric = _as_str(
        perf_raw.get("rouge_metric", "rouge-l"), ctx="router.performance.rouge_metric"
    )
    if rouge_metric not in ("rouge-1", "rouge-2", "rouge-l"):
        raise ValueError(
            "router.performance.rouge_metric must be one of: rouge-1, rouge-2, rouge-l; "
            f"got {rouge_metric!r}"
        )

    rouge_f1_threshold = _as_float(
        perf_raw.get("rouge_f1_threshold", 0.3),
        ctx="router.performance.rouge_f1_threshold",
    )
    perf_lowercase = _as_bool(
        perf_raw.get("lowercase", True), ctx="router.performance.lowercase"
    )

    if not (0.0 <= float(rouge_f1_threshold) <= 1.0):
        raise ValueError(
            "router.performance.rouge_f1_threshold must be in [0,1], "
            f"got {rouge_f1_threshold}"
        )

    detailed_raw = _as_mapping(
        router_raw.get("detailed_csv", {}), ctx="root.router.detailed_csv"
    )
    detailed_enabled = _as_bool(
        detailed_raw.get("enabled", False), ctx="router.detailed_csv.enabled"
    )
    detailed_path = _as_str(
        detailed_raw.get("path", "router_detailed_answers.csv"),
        ctx="router.detailed_csv.path",
    )
    detailed_append = _as_bool(
        detailed_raw.get("append", False), ctx="router.detailed_csv.append"
    )
    detailed_flush_every = _as_int(
        detailed_raw.get("flush_every", 1), ctx="router.detailed_csv.flush_every"
    )
    if detailed_flush_every < 1:
        raise ValueError(
            f"router.detailed_csv.flush_every must be >= 1, got {detailed_flush_every}"
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

        input_token_price = _as_float(
            bm.get("input_token_price", bm.get("input_price", 1.0)),
            ctx=f"backends[{i}].input_token_price",
        )
        cached_input_token_price = _as_float(
            bm.get(
                "cached_input_token_price",
                bm.get("cached_input_price", input_token_price),
            ),
            ctx=f"backends[{i}].cached_input_token_price",
        )
        output_token_price = _as_float(
            bm.get("output_token_price", bm.get("output_price", 1.0)),
            ctx=f"backends[{i}].output_token_price",
        )

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

        if input_token_price < 0:
            raise ValueError(
                f"backends[{i}].input_token_price must be >= 0, got {input_token_price}"
            )
        if cached_input_token_price < 0:
            raise ValueError(
                f"backends[{i}].cached_input_token_price must be >= 0, got {
                    cached_input_token_price
                }"
            )
        if output_token_price < 0:
            raise ValueError(
                f"backends[{i}].output_token_price must be >= 0, got {
                    output_token_price
                }"
            )

        backends.append(
            BackendConfig(
                backend_id=backend_id,
                base_url_v1=base_url_v1,
                api_key=api_key,
                model=model,
                capacity=int(capacity),
                input_token_price=float(input_token_price),
                cached_input_token_price=float(cached_input_token_price),
                output_token_price=float(output_token_price),
            )
        )

    if not backends:
        raise ValueError("Config must contain at least one backend in `backends:`")

    llmrouter_raw = _as_mapping(
        router_raw.get("llmrouter", {}), ctx="root.router.llmrouter"
    )
    llmrouter_name = _as_str(llmrouter_raw.get("name", ""), ctx="router.llmrouter.name")
    llmrouter_config_path = _as_str(
        llmrouter_raw.get("config_path", ""), ctx="router.llmrouter.config_path"
    )
    llmrouter_load_model_path = _as_str(
        llmrouter_raw.get("load_model_path", ""),
        ctx="router.llmrouter.load_model_path",
    )

    llmrouter_map_raw = _as_mapping(
        llmrouter_raw.get("model_name_to_backend_id", {}),
        ctx="router.llmrouter.model_name_to_backend_id",
    )
    llmrouter_map: dict[str, str] = {}
    for k, v in llmrouter_map_raw.items():
        llmrouter_map[
            _as_str(k, ctx="router.llmrouter.model_name_to_backend_id.key")
        ] = _as_str(v, ctx="router.llmrouter.model_name_to_backend_id.value")

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
            performance=RouterPerformanceConfig(
                enabled=bool(perf_enabled),
                evaluator=perf_evaluator,
                coqa_split=str(perf_coqa_split or "validation"),
                rouge_metric=str(rouge_metric or "rouge-l"),
                rouge_f1_threshold=float(rouge_f1_threshold),
                lowercase=bool(perf_lowercase),
            ),
            detailed_csv=RouterDetailedCsvConfig(
                enabled=bool(detailed_enabled),
                path=str(detailed_path or "router_detailed_answers.csv"),
                append=bool(detailed_append),
                flush_every=int(detailed_flush_every),
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
            llmrouter=RouterLLMRouterConfig(
                name=str(llmrouter_name),
                config_path=str(llmrouter_config_path),
                load_model_path=str(llmrouter_load_model_path),
                model_name_to_backend_id=dict(llmrouter_map),
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
