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
    DefaultDict[int, List[float]],  # obs_cache_ratio
    DefaultDict[int, List[float]],  # pred_cache_ratio
    DefaultDict[int, List[float]],  # obs_latency_ms
    DefaultDict[int, List[float]],  # pred_cost_tokens
    DefaultDict[int, List[float]],  # obs_cost_tokens
    DefaultDict[int, List[float]],  # obs_total_tokens
    DefaultDict[int, List[float]],  # pred_perf_prob
    DefaultDict[int, List[float]],  # correct (0/1)
    DefaultDict[int, List[int]],  # obs_prompt_tokens
    DefaultDict[int, List[int]],  # obs_cached_tokens
    DefaultDict[int, List[float]],  # pred_welfare
    DefaultDict[int, List[float]],  # obs_welfare
    DefaultDict[int, List[float]],  # vcg_fee
    DefaultDict[int, List[float]],  # vcg_total_payment
    DefaultDict[int, List[float]],  # auction_matched (0/1)
]:
    by_turn_obs_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_latency: DefaultDict[int, List[float]] = defaultdict(list)

    by_turn_pred_cost: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_cost: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_total: DefaultDict[int, List[float]] = defaultdict(list)

    by_turn_pred_perf: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_correct: DefaultDict[int, List[float]] = defaultdict(list)

    by_turn_prompt_tok: DefaultDict[int, List[int]] = defaultdict(list)
    by_turn_cached_tok: DefaultDict[int, List[int]] = defaultdict(list)

    by_turn_pred_welfare: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_welfare: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_vcg_fee: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_vcg_pay: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_matched: DefaultDict[int, List[float]] = defaultdict(list)

    for s in dialogue_series:
        for (
            t,
            ocr,
            pcr,
            lat,
            pcost,
            ocost,
            otok,
            pperf,
            corr,
            pt,
            ct,
            pw,
            ow,
            fee,
            pay,
            matched,
        ) in zip(
            s.turns,
            s.obs_cache_ratio,
            s.pred_cache_ratio,
            s.obs_latency_ms,
            s.pred_cost_tokens,
            s.obs_cost_tokens,
            [float(x) for x in s.obs_total_tokens],
            s.pred_perf_prob,
            s.correct,
            s.obs_prompt_tokens,
            s.obs_cached_tokens,
            s.pred_welfare,
            s.obs_welfare,
            s.vcg_fee,
            s.vcg_total_payment,
            s.auction_matched,
        ):
            if _is_finite(ocr):
                by_turn_obs_cache[t].append(float(ocr))
            if _is_finite(pcr):
                by_turn_pred_cache[t].append(float(pcr))
            if _is_finite(lat):
                by_turn_latency[t].append(float(lat))

            if _is_finite(pcost):
                by_turn_pred_cost[t].append(float(pcost))
            if _is_finite(ocost):
                by_turn_obs_cost[t].append(float(ocost))
            if _is_finite(otok):
                by_turn_obs_total[t].append(float(otok))

            if _is_finite(pperf):
                by_turn_pred_perf[t].append(float(pperf))
            by_turn_correct[t].append(1.0 if bool(corr) else 0.0)

            by_turn_prompt_tok[t].append(int(pt))
            by_turn_cached_tok[t].append(int(ct))

            if _is_finite(pw):
                by_turn_pred_welfare[t].append(float(pw))
            if _is_finite(ow):
                by_turn_obs_welfare[t].append(float(ow))
            if _is_finite(fee):
                by_turn_vcg_fee[t].append(float(fee))
            if _is_finite(pay):
                by_turn_vcg_pay[t].append(float(pay))

            by_turn_matched[t].append(1.0 if bool(matched) else 0.0)

    return (
        by_turn_obs_cache,
        by_turn_pred_cache,
        by_turn_latency,
        by_turn_pred_cost,
        by_turn_obs_cost,
        by_turn_obs_total,
        by_turn_pred_perf,
        by_turn_correct,
        by_turn_prompt_tok,
        by_turn_cached_tok,
        by_turn_pred_welfare,
        by_turn_obs_welfare,
        by_turn_vcg_fee,
        by_turn_vcg_pay,
        by_turn_matched,
    )


def _print_per_turn_aggregates(
    *,
    by_turn_obs_cache: DefaultDict[int, List[float]],
    by_turn_pred_cache: DefaultDict[int, List[float]],
    by_turn_latency: DefaultDict[int, List[float]],
    by_turn_pred_cost: DefaultDict[int, List[float]],
    by_turn_obs_cost: DefaultDict[int, List[float]],
    by_turn_obs_total: DefaultDict[int, List[float]],
    by_turn_pred_perf: DefaultDict[int, List[float]],
    by_turn_correct: DefaultDict[int, List[float]],
    by_turn_prompt_tok: DefaultDict[int, List[int]],
    by_turn_cached_tok: DefaultDict[int, List[int]],
    by_turn_pred_welfare: DefaultDict[int, List[float]],
    by_turn_obs_welfare: DefaultDict[int, List[float]],
    by_turn_vcg_fee: DefaultDict[int, List[float]],
    by_turn_vcg_pay: DefaultDict[int, List[float]],
    by_turn_matched: DefaultDict[int, List[float]],
) -> None:
    turns_sorted = sorted(by_turn_latency.keys())
    if not turns_sorted:
        return

    print("\nPer-turn aggregates (canonical: last record per turn per dialogue)")
    print("---------------------------------------------------------------")
    print(
        "turn  n   mean_lat(ms)  p90_lat  "
        "mean_obs_cache  mean_pred_cache  "
        "mean_pred_cost  mean_obs_cost  mean_obs_total_tok  "
        "mean_pred_perf  mean_correct  "
        "mean_pred_welfare  mean_obs_welfare  "
        "mean_vcg_fee  mean_vcg_pay  mean_matched  "
        "mean_prompt_tok  mean_cached_tok"
    )

    for t in turns_sorted[:30]:
        lats = by_turn_latency[t]
        ocrs = by_turn_obs_cache[t]
        pcrs = by_turn_pred_cache[t]
        pcs = by_turn_pred_cost[t]
        ocs = by_turn_obs_cost[t]
        ots = by_turn_obs_total[t]
        pps = by_turn_pred_perf[t]
        cors = by_turn_correct[t]
        pws = by_turn_pred_welfare[t]
        ows = by_turn_obs_welfare[t]
        fees = by_turn_vcg_fee[t]
        pays = by_turn_vcg_pay[t]
        mats = by_turn_matched[t]
        pts = by_turn_prompt_tok[t]
        cts = by_turn_cached_tok[t]

        mean_prompt = mean([float(x) for x in pts])
        mean_cached = mean([float(x) for x in cts])

        print(
            f"{t:>4}  {len(lats):>3}  "
            f"{mean(lats):>11.1f}  {quantile(lats, 0.90):>7.1f}  "
            f"{mean(ocrs):>14.3f}  {mean(pcrs):>15.3f}  "
            f"{mean(pcs):>13.3f}  {mean(ocs):>12.3f}  {mean(ots):>16.1f}  "
            f"{mean(pps):>13.3f}  {mean(cors):>12.3f}  "
            f"{mean(pws):>16.3f}  {mean(ows):>15.3f}  "
            f"{mean(fees):>11.3f}  {mean(pays):>11.3f}  {mean(mats):>12.3f}  "
            f"{mean_prompt:>15.1f}  {mean_cached:>15.1f}"
        )

    if len(turns_sorted) > 30:
        print(f"... ({len(turns_sorted) - 30} more turns)")
