"""
immas.analysis.run_analyzer

Analyze a JSONL router run log.

This file now focuses on:
- CLI, loading/filtering
- computing summary metrics
- producing CSV artifacts
- calling outlier reporting and plotting modules
"""

from __future__ import annotations

import argparse
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Mapping

from immas.analysis.analyzer_aggregates import (
    _compute_per_turn_aggregates,
    _print_per_turn_aggregates,
)
from immas.analysis.analyzer_metrics import (
    compute_binary_prob_metrics,
    extract_perf_pairs_from_records,
)
from immas.analysis.analyzer_outliers import (
    _find_inconsistent_usage_cases,
    _print_inconsistent_usage_cases,
    _print_residual_outliers,
    _print_top_latency_outliers,
    _print_turn_number_gaps,
    _top_latency_outliers,
    _turn_number_gaps,
)
from immas.analysis.analyzer_plots import _write_plots
from immas.analysis.analyzer_reports import (
    _write_backend_summary_csv,
    _write_dialogue_summary_csv,
)
from immas.analysis.analyzer_series import (
    _safe_float_series,
    _safe_int_series,
    _build_dialogue_series,
)
from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import (
    mae,
    mean,
    pearsonr,
    quantile,
    r2_score,
    rmse,
    load_jsonl,
    _f,
    _i,
    _s,
    _pairs,
    _write_turns_csv,
    _is_finite,
    _pearsonr_finite,
)


def _print_backend_usage(ok_by_end: List[Mapping[str, Any]]) -> None:
    """
    Print backend usage summary.

    This answers "which backend is used" at a glance, including model/base_url context.
    """
    by_backend: dict[str, list[Mapping[str, Any]]] = {}
    for r in ok_by_end:
        bid = _s(r.get("backend_id")) or "backend_unknown"
        by_backend.setdefault(bid, []).append(r)

    total = sum(len(v) for v in by_backend.values())
    if total <= 0:
        return

    print("\nBackend usage")
    print("-------------")
    for bid in sorted(by_backend.keys()):
        rs = by_backend[bid]
        n = len(rs)
        frac = float(n) / float(total)
        model = _s(rs[0].get("model")) if rs else ""
        base_url = _s(rs[0].get("backend_base_url_v1")) if rs else ""
        print(
            f"{bid:>16}  n={n:>5}  frac={frac:>6.2%}  model={model}  base_url={
                base_url
            }"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get("RUN_LOG_PATH", "router_run.jsonl"))
    ap.add_argument("--outdir", default="run_analysis")
    ap.add_argument(
        "--run-id", default="", help="If provided, filter to a specific run_id."
    )
    ap.add_argument(
        "--dialogue-id",
        default="",
        help="If provided, filter to a specific dialogue_id.",
    )
    ap.add_argument("--topk", type=int, default=10, help="How many outliers to print.")
    ap.add_argument(
        "--max-dialogue-plots",
        type=int,
        default=25,
        help="How many per-dialogue trace plots to write.",
    )
    ap.add_argument(
        "--min-dialogue-turns",
        type=int,
        default=3,
        help="Minimum number of turns required to plot a dialogue trace.",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    records = load_jsonl(log_path)

    if args.run_id:
        records = [r for r in records if _s(r.get("run_id")) == args.run_id]
    if args.dialogue_id:
        records = [r for r in records if _s(r.get("dialogue_id")) == args.dialogue_id]

    ok = [r for r in records if not r.get("error")]
    err = [r for r in records if r.get("error")]

    print(f"Loaded: {len(records)} records   ok={len(ok)}   errors={len(err)}")

    if not ok:
        print("No successful records to analyze.")
        return

    # Completion-order time series (for model evolution / load effects)
    ok_by_end = sorted(ok, key=lambda r: _f(r.get("t_end_monotonic")))

    _print_backend_usage(ok_by_end)

    # Pairwise metrics (skips missing/unparseable)
    pred_lat, obs_lat = _pairs(ok_by_end, "pred_latency_ms", "obs_latency_ms")
    pred_cost, obs_cost = _pairs(ok_by_end, "pred_cost_tokens", "obs_total_tokens")
    pred_cache, obs_cache = _pairs(ok_by_end, "pred_cache_ratio", "obs_cache_ratio")

    # Performance pairs (probability vs label)
    pred_perf, obs_correct_bool = extract_perf_pairs_from_records(
        [dict(r) for r in ok_by_end]
    )

    # Plot-friendly series (NaN for missing)
    obs_cache_ratio_all = _safe_float_series(ok_by_end, "obs_cache_ratio")
    obs_prompt_tokens_all = _safe_int_series(ok_by_end, "obs_prompt_tokens")
    obs_cached_tokens_all = _safe_int_series(ok_by_end, "obs_cached_tokens")

    correct = [bool(r.get("correct", True)) for r in ok_by_end]
    acc = sum(1 for c in correct if c) / len(correct) if correct else 0.0

    obs_lat_finite = [x for x in obs_lat if _is_finite(x)]
    obs_cost_finite = [x for x in obs_cost if _is_finite(x)]

    print("\nPredictor outputs: accuracy / regression / calibration")
    print("------------------------------------------------------")
    print(f"Observed accuracy(correct):        {acc:.4f}")

    print(f"\nLatency pairs:                     n={len(obs_lat)}")
    print(f"Latency MAE (ms):                  {mae(pred_lat, obs_lat):.2f}")
    print(f"Latency RMSE (ms):                 {rmse(pred_lat, obs_lat):.2f}")
    print(f"Latency corr:                      {pearsonr(pred_lat, obs_lat):.3f}")
    print(f"Latency R^2:                       {r2_score(pred_lat, obs_lat):.3f}")
    if obs_lat_finite:
        print(
            "Latency quantiles (ms): "
            f"p50={quantile(obs_lat_finite, 0.50):.1f}  "
            f"p90={quantile(obs_lat_finite, 0.90):.1f}  "
            f"p99={quantile(obs_lat_finite, 0.99):.1f}"
        )

    print(f"\nCost pairs:                        n={len(obs_cost)}")
    print(f"Cost MAE (tok):                    {mae(pred_cost, obs_cost):.2f}")
    print(f"Cost RMSE (tok):                   {rmse(pred_cost, obs_cost):.2f}")
    print(f"Cost corr:                         {pearsonr(pred_cost, obs_cost):.3f}")
    print(f"Cost R^2:                          {r2_score(pred_cost, obs_cost):.3f}")
    if obs_cost_finite:
        print(
            "Cost quantiles (tok): "
            f"p50={quantile(obs_cost_finite, 0.50):.1f}  "
            f"p90={quantile(obs_cost_finite, 0.90):.1f}  "
            f"p99={quantile(obs_cost_finite, 0.99):.1f}"
        )

    # Cache reuse summary (no mismatch deep-dive)
    print("\nCache reuse (observed) + router proxy (pred_cache_ratio)")
    print("--------------------------------------------------------")
    print(
        f"Mean obs_prompt_tokens:            {
            mean([float(x) for x in obs_prompt_tokens_all]):.1f}"
    )
    print(
        f"Mean obs_cached_tokens:            {
            mean([float(x) for x in obs_cached_tokens_all]):.1f}"
    )

    obs_cr_finite = [x for x in obs_cache_ratio_all if _is_finite(x)]
    if obs_cr_finite:
        print(f"Mean obs_cache_ratio:              {mean(obs_cr_finite):.3f}")
        print(
            "obs_cache_ratio quantiles: "
            f"p50={quantile(obs_cr_finite, 0.50):.3f}  "
            f"p90={quantile(obs_cr_finite, 0.90):.3f}  "
            f"p99={quantile(obs_cr_finite, 0.99):.3f}"
        )
    else:
        print("Mean obs_cache_ratio:              0.000")

    pred_cr_finite = [x for x in pred_cache if _is_finite(x)]
    if pred_cr_finite:
        print(f"Mean pred_cache_ratio (proxy):     {mean(pred_cr_finite):.3f}")

    # Performance probability metrics
    perf_metrics = compute_binary_prob_metrics(pred_perf, obs_correct_bool)
    print("\nPerformance probability (pred_perf_prob)")
    print("---------------------------------------")
    print(f"Pairs:                             n={perf_metrics.n}")
    print(f"Mean pred_perf_prob:               {perf_metrics.mean_pred:.3f}")
    print(f"Mean observed correct rate:        {perf_metrics.mean_obs:.3f}")
    print(f"Accuracy at threshold 0.5:         {perf_metrics.accuracy_at_0_5:.3f}")
    print(f"Brier score:                       {perf_metrics.brier:.4f}")
    print(f"Log loss:                          {perf_metrics.log_loss:.4f}")

    obs_lat_all_plot = _safe_float_series(ok_by_end, "obs_latency_ms")
    cache_lat_corr = _pearsonr_finite(obs_cache_ratio_all, obs_lat_all_plot)
    print(f"\nObs corr(obs_cache_ratio, latency): {cache_lat_corr:.3f}")

    # Conversation-ordered debug CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ok_by_turn = sorted(
        ok,
        key=lambda r: (
            _s(r.get("dialogue_id")),
            _i(r.get("turn_number")),
            _f(r.get("t_start_monotonic")),
        ),
    )
    turns_csv = outdir / "turns_sorted.csv"
    _write_turns_csv(out_path=turns_csv, records=ok_by_turn)
    print(f"\nWrote conversation-ordered debug CSV: {turns_csv.resolve()}")

    # Group by dialogue_id
    by_did: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in ok_by_turn:
        by_did[_s(r.get("dialogue_id"))].append(r)

    # Turn continuity warnings
    gaps = _turn_number_gaps(by_did)
    _print_turn_number_gaps(gaps)

    # Outliers / diagnostics
    topk = int(max(1, args.topk))
    top_lat = _top_latency_outliers(ok_by_end, topk=topk)
    _print_top_latency_outliers(top_lat)

    _print_residual_outliers(ok_by_end, topk=topk)

    inconsistent = _find_inconsistent_usage_cases(ok_by_end)
    _print_inconsistent_usage_cases(inconsistent, topk=topk)

    # Build per-dialogue canonical series (turn-aligned)
    dialogue_series: List[DialogueSeries] = []
    for did, rs in by_did.items():
        s = _build_dialogue_series(dialogue_id=did, records=rs)
        if s is not None:
            dialogue_series.append(s)

    # Write dialogue summary CSV
    dialogue_summary_csv = outdir / "dialogue_summary.csv"
    _write_dialogue_summary_csv(out_path=dialogue_summary_csv, series=dialogue_series)
    print(f"\nWrote dialogue summary CSV: {dialogue_summary_csv.resolve()}")

    # Write backend summary CSV
    backend_summary_csv = outdir / "backend_summary.csv"
    _write_backend_summary_csv(out_path=backend_summary_csv, ok_by_end=ok_by_end)
    print(f"Wrote backend summary CSV: {backend_summary_csv.resolve()}")

    # Per-turn aggregates table
    (
        by_turn_obs_cache,
        by_turn_pred_cache,
        by_turn_latency,
        by_turn_pred_cost,
        by_turn_obs_total,
        by_turn_pred_perf,
        by_turn_correct,
        by_turn_prompt_tok,
        by_turn_cached_tok,
    ) = _compute_per_turn_aggregates(dialogue_series)

    _print_per_turn_aggregates(
        by_turn_obs_cache=by_turn_obs_cache,
        by_turn_pred_cache=by_turn_pred_cache,
        by_turn_latency=by_turn_latency,
        by_turn_pred_cost=by_turn_pred_cost,
        by_turn_obs_total=by_turn_obs_total,
        by_turn_pred_perf=by_turn_pred_perf,
        by_turn_correct=by_turn_correct,
        by_turn_prompt_tok=by_turn_prompt_tok,
        by_turn_cached_tok=by_turn_cached_tok,
    )

    _write_plots(
        outdir=outdir,
        ok_by_end=ok_by_end,
        pred_lat=pred_lat,
        obs_lat=obs_lat,
        pred_cost=pred_cost,
        obs_cost=obs_cost,
        pred_cache=pred_cache,
        obs_cache=obs_cache,
        dialogue_series=dialogue_series,
        top_lat=top_lat,
        max_dialogue_plots=int(max(0, args.max_dialogue_plots)),
        min_dialogue_turns=int(max(1, args.min_dialogue_turns)),
        topk=topk,
    )


if __name__ == "__main__":
    main()
