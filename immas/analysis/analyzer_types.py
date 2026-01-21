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
    obs_total_tokens: List[int]

    pred_perf_prob: List[float]
    correct: List[bool]

    obs_prompt_tokens: List[int]
    obs_cached_tokens: List[int]

    prompt_chars: List[int]
    cached_prompt_chars: List[int]
    kvmatch_lcp_chars: List[int]
