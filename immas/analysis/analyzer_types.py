"""
immas.analysis.analyzer_types

Type definitions used by the analyzer.

Separated to avoid circular imports and keep run_analyzer.py small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class DialogueSeries:
    """
    Canonical per-turn series for one dialogue (one record per turn_number).

    Notes
    -----
    This structure intentionally contains both:
    - router inputs/observations (e.g. obs_*),
    - predictor outputs (pred_*),
    - routing identity (backend_id/model),
    so that per-dialogue analysis can answer:
    - which backend served which turn,
    - how predictions behaved across turns,
    - whether backend switching affects cache/latency/cost/performance.

    Cost
    ----
    `pred_cost_tokens` is the router's cost proxy prediction (arbitrary units).
    `obs_cost_tokens` is derived during analysis from logged usage fields using
    per-backend token prices (when available).

    Welfare
    ------
    `pred_welfare` is the predicted decision-time welfare in the router's welfare unit.
    `obs_welfare` is a post-hoc welfare computed from observed latency/cost and the
    `correct` label as a proxy for realized quality.
    """

    dialogue_id: str
    turns: List[int]

    backend_id: List[str]
    model: List[str]
    source: List[str]

    obs_latency_ms: List[float]
    pred_latency_ms: List[float]

    obs_cache_ratio: List[float]
    pred_cache_ratio: List[float]
    kvmatch_text: List[float]

    pred_cost_tokens: List[float]
    obs_cost_tokens: List[float]
    obs_total_tokens: List[int]

    pred_perf_prob: List[float]
    correct: List[bool]

    # Welfare series (analysis-derived, but based on router config params).
    pred_client_valuation: List[float]
    pred_base_cost: List[float]
    pred_welfare: List[float]

    obs_client_valuation: List[float]
    obs_base_cost: List[float]
    obs_welfare: List[float]

    auction_matched: List[bool]
    vcg_fee: List[float]
    vcg_total_payment: List[float]

    obs_prompt_tokens: List[int]
    obs_cached_tokens: List[int]

    prompt_chars: List[int]
    cached_prompt_chars: List[int]
    kvmatch_lcp_chars: List[int]
