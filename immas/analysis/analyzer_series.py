"""
immas.analysis.analyzer_series

Helpers for:
- finite filtering for correlation/scatter plots
- extracting plot-friendly series with NaNs/zeros
- turning a dialogue's raw records into a canonical per-turn DialogueSeries

Logic is identical to the original run_analyzer.py, just moved here.
"""

from __future__ import annotations

import hashlib
import math
import re

from typing import Any, Dict, List, Mapping, Optional, Sequence

from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import _f, _i, _s


def _safe_float_series(records: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    """
    Extract a float series from records. Missing/unparseable values become NaN.

    Intended for plotting (where NaNs are acceptable).
    """

    out: List[float] = []
    for r in records:
        v = r.get(key)
        if v is None:
            out.append(math.nan)
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(math.nan)
    return out


def _safe_int_series(records: Sequence[Mapping[str, Any]], key: str) -> List[int]:
    """
    Extract an int series from records. Missing/unparseable values become 0.
    """

    out: List[int] = []
    for r in records:
        v = r.get(key)
        if v is None:
            out.append(0)
            continue
        try:
            out.append(int(v))
        except Exception:
            out.append(0)
    return out


def _sanitize_file_stem(dialogue_id: str, *, max_len: int = 80) -> str:
    """
    Make a filesystem-safe, mostly-stable filename stem from dialogue_id.

    We keep a readable prefix and append a short hash to avoid collisions.
    """

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", dialogue_id).strip("_")
    cleaned = cleaned[:max_len] if cleaned else "dialogue"
    h = hashlib.sha1(dialogue_id.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}__{h}"


def _reduce_last_record_per_turn(
    records: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """
    Reduce a dialogue's records to one record per turn_number.

    If retries exist (duplicate turn_number), keep the last one by:
    (turn_number, t_end_monotonic).
    """

    by_turn: Dict[int, Mapping[str, Any]] = {}
    rs = sorted(
        records,
        key=lambda r: (_i(r.get("turn_number")), _f(r.get("t_end_monotonic"))),
    )
    for r in rs:
        t = _i(r.get("turn_number"))
        if t <= 0:
            continue
        by_turn[t] = r
    return [by_turn[t] for t in sorted(by_turn.keys())]


def _build_dialogue_series(
    *, dialogue_id: str, records: Sequence[Mapping[str, Any]]
) -> Optional[DialogueSeries]:
    """
    Build a canonical per-turn series for plotting/aggregation.

    Returns None if no valid (turn_number >= 1) records exist.
    """

    per_turn = _reduce_last_record_per_turn(records)
    if not per_turn:
        return None

    turns = [_i(r.get("turn_number")) for r in per_turn]

    return DialogueSeries(
        dialogue_id=dialogue_id,
        turns=turns,
        backend_id=[_s(r.get("backend_id")) for r in per_turn],
        model=[_s(r.get("model")) for r in per_turn],
        source=[_s(r.get("source")) for r in per_turn],
        obs_latency_ms=[_f(r.get("obs_latency_ms"), math.nan) for r in per_turn],
        pred_latency_ms=[_f(r.get("pred_latency_ms"), math.nan) for r in per_turn],
        obs_cache_ratio=[_f(r.get("obs_cache_ratio"), math.nan) for r in per_turn],
        pred_cache_ratio=[_f(r.get("pred_cache_ratio"), math.nan) for r in per_turn],
        kvmatch_text=[_f(r.get("kvmatch_text"), math.nan) for r in per_turn],
        pred_cost_tokens=[_f(r.get("pred_cost_tokens"), math.nan) for r in per_turn],
        obs_cost_tokens=[_f(r.get("obs_cost_tokens"), math.nan) for r in per_turn],
        obs_total_tokens=[_i(r.get("obs_total_tokens")) for r in per_turn],
        pred_perf_prob=[_f(r.get("pred_perf_prob"), math.nan) for r in per_turn],
        correct=[bool(r.get("correct", True)) for r in per_turn],
        pred_client_valuation=[
            _f(r.get("pred_client_valuation"), math.nan) for r in per_turn
        ],
        pred_base_cost=[_f(r.get("pred_base_cost"), math.nan) for r in per_turn],
        pred_welfare=[_f(r.get("pred_welfare"), math.nan) for r in per_turn],
        obs_client_valuation=[
            _f(r.get("obs_client_valuation"), math.nan) for r in per_turn
        ],
        obs_base_cost=[_f(r.get("obs_base_cost"), math.nan) for r in per_turn],
        obs_welfare=[_f(r.get("obs_welfare"), math.nan) for r in per_turn],
        auction_matched=[bool(r.get("auction_matched", False)) for r in per_turn],
        vcg_fee=[_f(r.get("vcg_fee"), math.nan) for r in per_turn],
        vcg_total_payment=[_f(r.get("vcg_total_payment"), math.nan) for r in per_turn],
        obs_prompt_tokens=[_i(r.get("obs_prompt_tokens")) for r in per_turn],
        obs_cached_tokens=[_i(r.get("obs_cached_tokens")) for r in per_turn],
        prompt_chars=[_i(r.get("prompt_chars")) for r in per_turn],
        cached_prompt_chars=[_i(r.get("cached_prompt_chars")) for r in per_turn],
        kvmatch_lcp_chars=[_i(r.get("kvmatch_lcp_chars")) for r in per_turn],
    )
