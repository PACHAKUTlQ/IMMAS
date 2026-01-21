"""
immas.analysis.analyzer_aggregates

Per-turn aggregates across dialogues using the canonical per-turn series.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Sequence

from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import mean, quantile, _is_finite


def _compute_per_turn_aggregates(
    dialogue_series: Sequence[DialogueSeries],
) -> tuple[
    DefaultDict[int, List[float]],
    DefaultDict[int, List[float]],
    DefaultDict[int, List[float]],
    DefaultDict[int, List[int]],
    DefaultDict[int, List[int]],
]:
    by_turn_obs_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_latency: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_prompt_tok: DefaultDict[int, List[int]] = defaultdict(list)
    by_turn_cached_tok: DefaultDict[int, List[int]] = defaultdict(list)

    for s in dialogue_series:
        for t, ocr, pcr, lat, pt, ct in zip(
            s.turns,
            s.obs_cache_ratio,
            s.pred_cache_ratio,
            s.obs_latency_ms,
            s.obs_prompt_tokens,
            s.obs_cached_tokens,
        ):
            if _is_finite(ocr):
                by_turn_obs_cache[t].append(float(ocr))
            if _is_finite(pcr):
                by_turn_pred_cache[t].append(float(pcr))
            if _is_finite(lat):
                by_turn_latency[t].append(float(lat))
            by_turn_prompt_tok[t].append(int(pt))
            by_turn_cached_tok[t].append(int(ct))

    return (
        by_turn_obs_cache,
        by_turn_pred_cache,
        by_turn_latency,
        by_turn_prompt_tok,
        by_turn_cached_tok,
    )


def _print_per_turn_aggregates(
    *,
    by_turn_obs_cache: DefaultDict[int, List[float]],
    by_turn_pred_cache: DefaultDict[int, List[float]],
    by_turn_latency: DefaultDict[int, List[float]],
    by_turn_prompt_tok: DefaultDict[int, List[int]],
    by_turn_cached_tok: DefaultDict[int, List[int]],
) -> None:
    turns_sorted = sorted(by_turn_latency.keys())
    if not turns_sorted:
        return

    print("\nPer-turn aggregates (canonical: last record per turn per dialogue)")
    print("---------------------------------------------------------------")
    print(
        "turn  n   mean_lat(ms)  p90_lat  mean_obs_cache  mean_pred_cache  mean_prompt_tok  mean_cached_tok"
    )
    for t in turns_sorted[:30]:
        lats = by_turn_latency[t]
        ocrs = by_turn_obs_cache[t]
        pcrs = by_turn_pred_cache[t]
        pts = by_turn_prompt_tok[t]
        cts = by_turn_cached_tok[t]
        print(
            f"{t:>4}  {len(lats):>3}  "
            f"{mean(lats):>11.1f}  {quantile(lats, 0.90):>7.1f}  "
            f"{mean(ocrs):>14.3f}  {mean(pcrs):>15.3f}  "
            f"{mean([float(x) for x in pts]):>15.1f}  {
                mean([float(x) for x in cts]):>15.1f}"
        )
    if len(turns_sorted) > 30:
        print(f"... ({len(turns_sorted) - 30} more turns)")
