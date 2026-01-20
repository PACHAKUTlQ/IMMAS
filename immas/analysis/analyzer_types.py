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
    """Canonical per-turn series for one dialogue (one record per turn_number)."""

    dialogue_id: str
    turns: List[int]

    obs_latency_ms: List[float]
    pred_latency_ms: List[float]

    obs_cache_ratio: List[float]
    pred_cache_ratio: List[float]
    kvmatch_text: List[float]

    obs_prompt_tokens: List[int]
    obs_cached_tokens: List[int]

    prompt_chars: List[int]
    cached_prompt_chars: List[int]
    kvmatch_lcp_chars: List[int]
