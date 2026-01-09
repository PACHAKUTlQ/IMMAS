"""
vLLM-ish latency simulation driven by server load.

Design goals
------------
- Simple, explainable model that reproduces: "mostly flat baseline + rare batch spikes".
- Works now (single client), but becomes meaningful once you add parallel requests.
- Structured for extension: later you can add TTFT vs decode decomposition, queueing,
  KV-cache reuse, per-model speed, etc.

Core idea (inferred from the provided figures)
---------------------------------------------
- Baseline TTFT stays roughly constant under load.
- Occasionally, entire batches experience a stall (large TTFT spike).
- Stall probability increases as utilization approaches a threshold.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from load_tracker import LoadSnapshot


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    return float(v)


@dataclass(frozen=True, slots=True)
class VllmLatencySimConfig:
    """
    Configuration for the latency simulator.

    Environment variables (all optional)
    -----------------------------------
    FAKE_VLLM_SIM_ENABLED
    FAKE_VLLM_CAPACITY
    FAKE_VLLM_BACKGROUND_INFLIGHT
    FAKE_VLLM_BASE_TTFT_OVERHEAD_S
    FAKE_VLLM_PREFILL_TPS
    FAKE_VLLM_DECODE_TPS
    FAKE_VLLM_TTFT_LOAD_ALPHA
    FAKE_VLLM_TTFT_LOAD_POWER
    FAKE_VLLM_DECODE_LOAD_ALPHA
    FAKE_VLLM_DECODE_LOAD_POWER
    FAKE_VLLM_QUEUE_ALPHA_S
    FAKE_VLLM_QUEUE_POWER
    FAKE_VLLM_STALL_LOAD_THRESHOLD
    FAKE_VLLM_STALL_START_PROB_MAX
    FAKE_VLLM_STALL_PROB_POWER
    FAKE_VLLM_STALL_LEN_MIN
    FAKE_VLLM_STALL_LEN_MAX
    FAKE_VLLM_STALL_EXTRA_LOGN_MU
    FAKE_VLLM_STALL_EXTRA_LOGN_SIGMA
    FAKE_VLLM_STALL_EXTRA_MAX_S
    FAKE_VLLM_WARMUP_REQUESTS
    FAKE_VLLM_WARMUP_EXTRA_S
    FAKE_VLLM_JITTER_STD_S
    FAKE_VLLM_MAX_TOTAL_S
    FAKE_VLLM_SEED
    """

    enabled: bool = True

    # Load model
    capacity: int = 32
    background_inflight: int = 0

    # Baseline latency components
    base_ttft_overhead_s: float = 0.010
    prefill_tps: float = 60000.0
    decode_tps: float = 2500.0

    # Smooth load scaling (small effect, matches "mostly flat baseline")
    ttft_load_alpha: float = 0.25
    ttft_load_power: float = 2.0
    decode_load_alpha: float = 0.25
    decode_load_power: float = 2.0

    # Overload queue term (only when utilization > 1)
    queue_alpha_s: float = 0.10
    queue_power: float = 2.0

    # Stall model (rare spikes affecting a run of requests)
    stall_load_threshold: float = 0.85  # u0 in the description
    stall_start_prob_max: float = 0.0015  # p_max at utilization ~= 1.0
    stall_prob_power: float = 2.0
    stall_len_min: int = 20
    stall_len_max: int = 60

    # Stall extra delay distribution: LogNormal(mu, sigma), clipped
    # Defaults chosen so mean ~ 0.25s and tail can reach ~0.6s
    stall_extra_logn_mu: float = -1.47
    stall_extra_logn_sigma: float = 0.4
    stall_extra_max_s: float = 0.6

    # Warmup spike (optional)
    warmup_requests: int = 0
    warmup_extra_s: float = 0.35

    # Noise/jitter + clamp
    jitter_std_s: float = 0.002
    max_total_s: float = 5.0

    seed: int = 0

    @classmethod
    def from_env(cls) -> "VllmLatencySimConfig":
        return cls(
            enabled=_env_bool("FAKE_VLLM_SIM_ENABLED", True),
            capacity=_env_int("FAKE_VLLM_CAPACITY", 32),
            background_inflight=_env_int("FAKE_VLLM_BACKGROUND_INFLIGHT", 0),
            base_ttft_overhead_s=_env_float("FAKE_VLLM_BASE_TTFT_OVERHEAD_S", 0.010),
            prefill_tps=_env_float("FAKE_VLLM_PREFILL_TPS", 60000.0),
            decode_tps=_env_float("FAKE_VLLM_DECODE_TPS", 2500.0),
            ttft_load_alpha=_env_float("FAKE_VLLM_TTFT_LOAD_ALPHA", 0.25),
            ttft_load_power=_env_float("FAKE_VLLM_TTFT_LOAD_POWER", 2.0),
            decode_load_alpha=_env_float("FAKE_VLLM_DECODE_LOAD_ALPHA", 0.25),
            decode_load_power=_env_float("FAKE_VLLM_DECODE_LOAD_POWER", 2.0),
            queue_alpha_s=_env_float("FAKE_VLLM_QUEUE_ALPHA_S", 0.10),
            queue_power=_env_float("FAKE_VLLM_QUEUE_POWER", 2.0),
            stall_load_threshold=_env_float("FAKE_VLLM_STALL_LOAD_THRESHOLD", 0.85),
            stall_start_prob_max=_env_float("FAKE_VLLM_STALL_START_PROB_MAX", 0.0015),
            stall_prob_power=_env_float("FAKE_VLLM_STALL_PROB_POWER", 2.0),
            stall_len_min=_env_int("FAKE_VLLM_STALL_LEN_MIN", 20),
            stall_len_max=_env_int("FAKE_VLLM_STALL_LEN_MAX", 60),
            stall_extra_logn_mu=_env_float("FAKE_VLLM_STALL_EXTRA_LOGN_MU", -1.47),
            stall_extra_logn_sigma=_env_float("FAKE_VLLM_STALL_EXTRA_LOGN_SIGMA", 0.4),
            stall_extra_max_s=_env_float("FAKE_VLLM_STALL_EXTRA_MAX_S", 0.6),
            warmup_requests=_env_int("FAKE_VLLM_WARMUP_REQUESTS", 0),
            warmup_extra_s=_env_float("FAKE_VLLM_WARMUP_EXTRA_S", 0.35),
            jitter_std_s=_env_float("FAKE_VLLM_JITTER_STD_S", 0.002),
            max_total_s=_env_float("FAKE_VLLM_MAX_TOTAL_S", 5.0),
            seed=_env_int("FAKE_VLLM_SEED", 0),
        )


@dataclass(frozen=True, slots=True)
class SimulatedLatency:
    """
    Decomposed latency, in seconds.

    This keeps TTFT separate so you can later implement streaming:
    - sleep TTFT
    - then stream tokens according to decode_s
    """

    ttft_s: float
    decode_s: float
    queue_s: float
    stall_s: float
    warmup_s: float
    total_s: float

    effective_inflight: int
    utilization: float
    rps: float


class VllmLatencySimulator:
    """
    Stateful latency simulator.

    State
    -----
    - Stall events persist for a run of N subsequent requests.
    - Warmup applies for the first `warmup_requests` calls.
    """

    def __init__(self, cfg: VllmLatencySimConfig) -> None:
        self._cfg = cfg
        self._rng = random.Random(cfg.seed)

        self._req_count: int = 0

        # Stall state
        self._stall_remaining: int = 0
        self._stall_extra_s: float = 0.0

        self._validate_cfg()

    @property
    def cfg(self) -> VllmLatencySimConfig:
        return self._cfg

    def _validate_cfg(self) -> None:
        c = self._cfg
        if c.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {c.capacity}")
        if c.prefill_tps <= 0:
            raise ValueError(f"prefill_tps must be > 0, got {c.prefill_tps}")
        if c.decode_tps <= 0:
            raise ValueError(f"decode_tps must be > 0, got {c.decode_tps}")
        if c.stall_len_min <= 0 or c.stall_len_max <= 0:
            raise ValueError("stall_len_min/max must be > 0")
        if c.stall_len_min > c.stall_len_max:
            raise ValueError("stall_len_min must be <= stall_len_max")
        if not (0.0 <= c.stall_load_threshold < 1.0):
            raise ValueError("stall_load_threshold must be in [0, 1)")
        if c.stall_start_prob_max < 0.0:
            raise ValueError("stall_start_prob_max must be >= 0")
        if c.max_total_s <= 0.0:
            raise ValueError("max_total_s must be > 0")

    def _compute_stall_delay(self, *, u_capped: float) -> float:
        """
        If already stalled: consume one unit and return stall extra delay.
        Else: maybe start a stall depending on utilization.
        """
        if self._stall_remaining > 0:
            self._stall_remaining -= 1
            return self._stall_extra_s

        c = self._cfg
        if u_capped <= c.stall_load_threshold:
            return 0.0

        # Map utilization u in (threshold..1] to x in (0..1]
        denom = max(1e-9, 1.0 - c.stall_load_threshold)
        x = (u_capped - c.stall_load_threshold) / denom
        x = max(0.0, min(1.0, x))

        p = c.stall_start_prob_max * (x**c.stall_prob_power)
        if self._rng.random() >= p:
            return 0.0

        # Start a stall affecting a run of requests (batch-like correlation).
        length = self._rng.randint(c.stall_len_min, c.stall_len_max)
        self._stall_remaining = max(0, length - 1)

        extra = self._rng.lognormvariate(
            c.stall_extra_logn_mu, c.stall_extra_logn_sigma
        )
        extra = min(float(extra), c.stall_extra_max_s)
        self._stall_extra_s = max(0.0, extra)
        return self._stall_extra_s

    def simulate(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        load: LoadSnapshot,
    ) -> SimulatedLatency:
        """
        Simulate TTFT and total latency based on token sizes + load.

        Parameters
        ----------
        prompt_tokens
            Estimated prompt tokens.
        completion_tokens
            Estimated completion tokens.
        load
            Current load snapshot (in-flight + recent RPS).

        Returns
        -------
        SimulatedLatency
            Decomposed latency in seconds.
        """
        if not self._cfg.enabled:
            return SimulatedLatency(
                ttft_s=0.0,
                decode_s=0.0,
                queue_s=0.0,
                stall_s=0.0,
                warmup_s=0.0,
                total_s=0.0,
                effective_inflight=max(1, load.inflight_requests),
                utilization=float(max(1, load.inflight_requests))
                / float(self._cfg.capacity),
                rps=float(load.rps),
            )

        self._req_count += 1

        p = max(0, int(prompt_tokens))
        cpl = max(0, int(completion_tokens))

        effective_inflight = max(
            1, int(load.inflight_requests) + max(0, self._cfg.background_inflight)
        )
        utilization = float(effective_inflight) / float(self._cfg.capacity)

        # Within-capacity effects are mild; overload handled separately by queue term.
        u_capped = min(utilization, 1.0)

        # Base TTFT ~ overhead + prefill compute
        base_ttft = self._cfg.base_ttft_overhead_s + (float(p) / self._cfg.prefill_tps)
        ttft = base_ttft * (
            1.0 + self._cfg.ttft_load_alpha * (u_capped**self._cfg.ttft_load_power)
        )

        # Decode time (post-first-token)
        decode = (float(cpl) / self._cfg.decode_tps) * (
            1.0 + self._cfg.decode_load_alpha * (u_capped**self._cfg.decode_load_power)
        )

        # Overload queue penalty (applies before TTFT)
        overload = max(0.0, utilization - 1.0)
        queue_s = self._cfg.queue_alpha_s * (overload**self._cfg.queue_power)

        # Warmup penalty
        warmup_s = (
            self._cfg.warmup_extra_s
            if self._req_count <= self._cfg.warmup_requests
            else 0.0
        )

        # Batch-correlated stall penalty (rare)
        stall_s = self._compute_stall_delay(u_capped=u_capped)

        ttft_s = max(0.0, ttft + queue_s + warmup_s + stall_s)

        jitter = (
            self._rng.normalvariate(0.0, self._cfg.jitter_std_s)
            if self._cfg.jitter_std_s > 0
            else 0.0
        )
        total = max(0.0, ttft_s + decode + jitter)
        total = min(total, self._cfg.max_total_s)

        return SimulatedLatency(
            ttft_s=ttft_s,
            decode_s=max(0.0, decode),
            queue_s=max(0.0, queue_s),
            stall_s=max(0.0, stall_s),
            warmup_s=max(0.0, warmup_s),
            total_s=total,
            effective_inflight=effective_inflight,
            utilization=utilization,
            rps=float(load.rps),
        )
