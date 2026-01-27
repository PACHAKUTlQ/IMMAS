"""
immas.analysis.analyzer_plots

All matplotlib plotting for the analyzer.

This module is intentionally isolated so that:
- the main analyzer can run without matplotlib
- the plotting code doesn't dominate run_analyzer.py
"""

from __future__ import annotations

import math

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Mapping, Sequence

from immas.analysis.analyzer_bins import _binned_means
from immas.analysis.analyzer_metrics import extract_perf_pairs_from_records
from immas.analysis.analyzer_series import (
    _safe_float_series,
    _safe_int_series,
    _sanitize_file_stem,
)
from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import _s, quantile, _is_finite, _pearsonr_finite


def _write_plots(
    *,
    outdir: Path,
    ok_by_end: Sequence[Mapping[str, Any]],
    pred_lat: Sequence[float],
    obs_lat: Sequence[float],
    pred_cost: Sequence[float],
    obs_cost: Sequence[float],
    pred_cache: Sequence[float],
    obs_cache: Sequence[float],
    pred_welfare: Sequence[float],
    obs_welfare: Sequence[float],
    dialogue_series: Sequence[DialogueSeries],
    top_lat: Sequence[Mapping[str, Any]],
    max_dialogue_plots: int,
    min_dialogue_turns: int,
    topk: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available; skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = outdir / "dialogues"
    dialogues_dir.mkdir(parents=True, exist_ok=True)

    # Plot-friendly series (NaN for missing)
    obs_lat_all_plot = _safe_float_series(ok_by_end, "obs_latency_ms")
    pred_lat_all_plot = _safe_float_series(ok_by_end, "pred_latency_ms")

    obs_cache_ratio_all = _safe_float_series(ok_by_end, "obs_cache_ratio")
    pred_cache_ratio_all = _safe_float_series(ok_by_end, "pred_cache_ratio")

    obs_prompt_tokens_all = _safe_int_series(ok_by_end, "obs_prompt_tokens")
    obs_cached_tokens_all = _safe_int_series(ok_by_end, "obs_cached_tokens")

    pred_perf_prob_all = _safe_float_series(ok_by_end, "pred_perf_prob")
    correct_all: list[bool] = [bool(r.get("correct", True)) for r in ok_by_end]
    correct_all_float: list[float] = [1.0 if c else 0.0 for c in correct_all]

    obs_welfare_all_plot = _safe_float_series(ok_by_end, "obs_welfare")
    pred_welfare_all_plot = _safe_float_series(ok_by_end, "pred_welfare")

    vcg_fee_all_plot = _safe_float_series(ok_by_end, "vcg_fee")
    vcg_total_payment_all_plot = _safe_float_series(ok_by_end, "vcg_total_payment")

    # Backend usage (categorical)
    backend_ids: list[str] = [
        _s(r.get("backend_id")) or "backend_unknown" for r in ok_by_end
    ]
    backend_counts: dict[str, int] = {}
    for bid in backend_ids:
        backend_counts[bid] = backend_counts.get(bid, 0) + 1

    obs_lat_finite: list[float] = [x for x in obs_lat if _is_finite(x)]
    obs_cr_finite: list[float] = [x for x in obs_cache_ratio_all if _is_finite(x)]
    obs_cost_finite: list[float] = [x for x in obs_cost if _is_finite(x)]

    obs_w_finite: list[float] = [x for x in obs_welfare if _is_finite(x)]
    pred_w_finite: list[float] = [x for x in pred_welfare if _is_finite(x)]

    # completion-order evolution plots
    xs_all: list[int] = list(range(len(ok_by_end)))

    # Time series: latency
    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_lat_all_plot, label="observed latency (ms)", linewidth=1.5)
    plt.plot(
        xs_all,
        pred_lat_all_plot,
        label="predicted latency (ms)",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.title("Latency over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "latency_timeseries.png", dpi=160)
    plt.close()

    # Time series: cost (cost proxy units)
    obs_cost_all_plot = _safe_float_series(ok_by_end, "obs_cost_tokens")
    pred_cost_all_plot = _safe_float_series(ok_by_end, "pred_cost_tokens")

    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_cost_all_plot, label="observed cost proxy", linewidth=1.5)
    plt.plot(
        xs_all,
        pred_cost_all_plot,
        label="predicted cost proxy (pred_cost_tokens)",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.title("Cost proxy over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("cost proxy (arbitrary units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cost_timeseries.png", dpi=160)
    plt.close()

    # Time series: cache ratio
    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_cache_ratio_all, label="observed cache ratio", linewidth=1.5)
    plt.plot(
        xs_all,
        pred_cache_ratio_all,
        label="pred_cache_ratio (router prefix proxy)",
        linewidth=1.0,
        alpha=0.85,
    )
    plt.title("Cache reuse over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("ratio")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cache_ratio_timeseries.png", dpi=160)
    plt.close()

    # Time series: performance probability vs correctness
    plt.figure(figsize=(12, 5))
    plt.plot(
        xs_all, pred_perf_prob_all, label="pred_perf_prob", linewidth=1.5, alpha=0.9
    )
    plt.plot(xs_all, correct_all_float, label="correct (0/1)", linewidth=1.0, alpha=0.7)
    plt.title("Performance probability over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("probability / label")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "performance_timeseries.png", dpi=160)
    plt.close()

    # Time series: welfare (pred vs obs)
    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_welfare_all_plot, label="obs_welfare", linewidth=1.6)
    plt.plot(
        xs_all, pred_welfare_all_plot, label="pred_welfare", linewidth=1.1, alpha=0.85
    )
    plt.title("Welfare over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("welfare (utility units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "welfare_timeseries.png", dpi=160)
    plt.close()

    # Payment time series (often sparse)
    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, vcg_fee_all_plot, label="vcg_fee", linewidth=1.4, alpha=0.9)
    plt.plot(
        xs_all,
        vcg_total_payment_all_plot,
        label="vcg_total_payment",
        linewidth=1.2,
        alpha=0.85,
    )
    plt.title("VCG payments over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("payment (utility units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "vcg_payments_timeseries.png", dpi=160)
    plt.close()

    # Backend usage bar plot
    if backend_counts:
        labels = sorted(backend_counts.keys())
        counts = [backend_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, counts, alpha=0.85)
        plt.title("Backend usage (count of successful requests)")
        plt.xlabel("backend_id")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "backend_usage_bar.png", dpi=160)
        plt.close()

    # Scatter: predicted vs observed latency
    if pred_lat and obs_lat:
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_lat, obs_lat, s=10, alpha=0.6)
        lo = min(min(pred_lat), min(obs_lat))
        hi = max(max(pred_lat), max(obs_lat))
        plt.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5
        )
        plt.title("Predicted vs observed latency")
        plt.xlabel("predicted latency (ms)")
        plt.ylabel("observed latency (ms)")
        plt.tight_layout()
        plt.savefig(outdir / "latency_scatter.png", dpi=160)
        plt.close()

    # Scatter: predicted vs observed cost proxy
    if pred_cost and obs_cost:
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_cost, obs_cost, s=10, alpha=0.6)
        lo = min(min(pred_cost), min(obs_cost))
        hi = max(max(pred_cost), max(obs_cost))
        plt.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5
        )
        plt.title("Predicted vs observed cost proxy")
        plt.xlabel("pred_cost_tokens (cost proxy)")
        plt.ylabel("obs_cost_tokens (cost proxy)")
        plt.tight_layout()
        plt.savefig(outdir / "cost_scatter.png", dpi=160)
        plt.close()

    # Scatter: predicted vs observed welfare
    if pred_w_finite and obs_w_finite and len(pred_w_finite) == len(obs_w_finite):
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_w_finite, obs_w_finite, s=10, alpha=0.6)
        lo = min(min(pred_w_finite), min(obs_w_finite))
        hi = max(max(pred_w_finite), max(obs_w_finite))
        plt.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5
        )
        plt.title("Predicted vs observed welfare")
        plt.xlabel("pred_welfare")
        plt.ylabel("obs_welfare")
        plt.tight_layout()
        plt.savefig(outdir / "welfare_scatter.png", dpi=160)
        plt.close()

    # Distributions
    if obs_lat_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(obs_lat_finite, bins=50, alpha=0.85)
        plt.title("Observed latency distribution")
        plt.xlabel("obs_latency_ms")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "obs_latency_hist.png", dpi=160)
        plt.close()

    if obs_cost_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(obs_cost_finite, bins=50, alpha=0.85)
        plt.title("Observed cost proxy distribution")
        plt.xlabel("obs_cost_tokens")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "obs_cost_hist.png", dpi=160)
        plt.close()

    if obs_cr_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(obs_cr_finite, bins=40, range=(0.0, 1.0), alpha=0.85)
        plt.title("Observed cache ratio distribution")
        plt.xlabel("obs_cache_ratio")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "obs_cache_ratio_hist.png", dpi=160)
        plt.close()

    if obs_w_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(obs_w_finite, bins=60, alpha=0.85)
        plt.title("Observed welfare distribution")
        plt.xlabel("obs_welfare")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "obs_welfare_hist.png", dpi=160)
        plt.close()

    # Payments distribution (matched tasks only usually)
    fee_finite = [x for x in vcg_fee_all_plot if _is_finite(float(x))]
    if fee_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(fee_finite, bins=50, alpha=0.85)
        plt.title("VCG fee distribution")
        plt.xlabel("vcg_fee")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "vcg_fee_hist.png", dpi=160)
        plt.close()

    # Performance distribution + calibration curve
    ps, ys = extract_perf_pairs_from_records([dict(r) for r in ok_by_end])
    if ps:
        plt.figure(figsize=(7, 5))
        plt.hist(ps, bins=30, range=(0.0, 1.0), alpha=0.85)
        plt.title("pred_perf_prob distribution")
        plt.xlabel("pred_perf_prob")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "pred_perf_prob_hist.png", dpi=160)
        plt.close()

    if ps and ys:
        ys_float = [1.0 if y else 0.0 for y in ys]
        centers, means_y, _counts = _binned_means(
            ps, ys_float, n_bins=20, x_min=0.0, x_max=1.0
        )
        if centers and means_y:
            plt.figure(figsize=(7, 5))
            plt.plot(
                centers,
                means_y,
                marker="o",
                linewidth=1.5,
                label="mean correct per pred bin",
            )
            plt.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                linestyle="--",
                linewidth=1,
                color="black",
                alpha=0.5,
                label="ideal",
            )
            plt.title("Performance calibration (reliability curve)")
            plt.xlabel("pred_perf_prob (binned)")
            plt.ylabel("mean correct")
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "performance_calibration_curve.png", dpi=160)
            plt.close()

    # Residual histograms: latency, cost, welfare
    if pred_lat and obs_lat:
        resid = [o - p for p, o in zip(pred_lat, obs_lat)]
        plt.figure(figsize=(7, 5))
        plt.hist(resid, bins=60, alpha=0.85)
        plt.title("Latency residuals distribution (obs - pred)")
        plt.xlabel("residual_ms")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "latency_residuals_hist.png", dpi=160)
        plt.close()

    if pred_cost and obs_cost:
        resid = [o - p for p, o in zip(pred_cost, obs_cost)]
        plt.figure(figsize=(7, 5))
        plt.hist(resid, bins=60, alpha=0.85)
        plt.title("Cost residuals distribution (obs - pred)")
        plt.xlabel("residual_cost")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "cost_residuals_hist.png", dpi=160)
        plt.close()

    if pred_w_finite and obs_w_finite and len(pred_w_finite) == len(obs_w_finite):
        resid = [o - p for p, o in zip(pred_w_finite, obs_w_finite)]
        plt.figure(figsize=(7, 5))
        plt.hist(resid, bins=60, alpha=0.85)
        plt.title("Welfare residuals distribution (obs - pred)")
        plt.xlabel("residual_welfare")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "welfare_residuals_hist.png", dpi=160)
        plt.close()

    # Cached tokens vs prompt tokens (sanity scatter)
    if obs_prompt_tokens_all and obs_cached_tokens_all:
        xs_pt: list[float] = []
        ys_ct: list[float] = []
        for pt, ct in zip(obs_prompt_tokens_all, obs_cached_tokens_all):
            if pt > 0 and ct >= 0:
                xs_pt.append(float(pt))
                ys_ct.append(float(ct))
        if xs_pt and ys_ct:
            plt.figure(figsize=(6, 6))
            plt.scatter(xs_pt, ys_ct, s=10, alpha=0.6)
            hi = max(xs_pt) if xs_pt else 1.0
            plt.plot(
                [0.0, hi],
                [0.0, hi],
                linestyle="--",
                linewidth=1,
                color="black",
                alpha=0.5,
            )
            plt.title("Observed cached_tokens vs prompt_tokens")
            plt.xlabel("obs_prompt_tokens")
            plt.ylabel("obs_cached_tokens")
            plt.tight_layout()
            plt.savefig(outdir / "cached_tokens_vs_prompt_tokens.png", dpi=160)
            plt.close()

    # Binned latency vs cache ratio
    centers, means_y, _counts = _binned_means(
        obs_cache_ratio_all, obs_lat_all_plot, n_bins=20, x_min=0.0, x_max=1.0
    )
    if centers and means_y:
        plt.figure(figsize=(7, 5))
        plt.plot(centers, means_y, marker="o", linewidth=1.5)
        plt.title("Binned mean latency vs observed cache ratio")
        plt.xlabel("obs_cache_ratio (binned)")
        plt.ylabel("mean obs_latency_ms")
        plt.xlim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(outdir / "binned_latency_vs_obs_cache_ratio.png", dpi=160)
        plt.close()

    # Within-dialogue / per-turn visualization (canonical per-turn)
    by_turn_obs_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_latency: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cost: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_cost: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_perf: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_correct: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_welfare: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_welfare: DefaultDict[int, List[float]] = defaultdict(list)

    for s in dialogue_series:
        for t, ocr, pcr, lat, pc, oc, pp, corr, pw, ow in zip(
            s.turns,
            s.obs_cache_ratio,
            s.pred_cache_ratio,
            s.obs_latency_ms,
            s.pred_cost_tokens,
            s.obs_cost_tokens,
            s.pred_perf_prob,
            s.correct,
            s.pred_welfare,
            s.obs_welfare,
        ):
            if _is_finite(ocr):
                by_turn_obs_cache[t].append(float(ocr))
            if _is_finite(pcr):
                by_turn_pred_cache[t].append(float(pcr))
            if _is_finite(lat):
                by_turn_latency[t].append(float(lat))
            if _is_finite(pc):
                by_turn_pred_cost[t].append(float(pc))
            if _is_finite(oc):
                by_turn_obs_cost[t].append(float(oc))
            if _is_finite(pp):
                by_turn_pred_perf[t].append(float(pp))
            by_turn_correct[t].append(1.0 if bool(corr) else 0.0)
            if _is_finite(pw):
                by_turn_pred_welfare[t].append(float(pw))
            if _is_finite(ow):
                by_turn_obs_welfare[t].append(float(ow))

    turns = sorted(by_turn_latency.keys())
    if turns:

        def _band(vals: Sequence[float]) -> tuple[float, float, float]:
            v = [float(x) for x in vals if _is_finite(float(x))]
            if not v:
                return (math.nan, math.nan, math.nan)
            return (quantile(v, 0.25), quantile(v, 0.50), quantile(v, 0.75))

        t_x: list[int] = []

        obs_cache_p25: list[float] = []
        obs_cache_p50: list[float] = []
        obs_cache_p75: list[float] = []

        pred_cache_p25: list[float] = []
        pred_cache_p50: list[float] = []
        pred_cache_p75: list[float] = []

        lat_p25: list[float] = []
        lat_p50: list[float] = []
        lat_p75: list[float] = []

        pred_cost_p50: list[float] = []
        obs_cost_p50: list[float] = []

        pred_perf_p50: list[float] = []
        correct_mean: list[float] = []

        pred_w_p50: list[float] = []
        obs_w_p50: list[float] = []

        for t in turns:
            t_x.append(t)

            a, b, c = _band(by_turn_obs_cache[t])
            obs_cache_p25.append(a)
            obs_cache_p50.append(b)
            obs_cache_p75.append(c)

            a, b, c = _band(by_turn_pred_cache[t])
            pred_cache_p25.append(a)
            pred_cache_p50.append(b)
            pred_cache_p75.append(c)

            a, b, c = _band(by_turn_latency[t])
            lat_p25.append(a)
            lat_p50.append(b)
            lat_p75.append(c)

            pred_cost_p50.append(
                quantile(by_turn_pred_cost[t], 0.50)
                if by_turn_pred_cost[t]
                else math.nan
            )
            obs_cost_p50.append(
                quantile(by_turn_obs_cost[t], 0.50) if by_turn_obs_cost[t] else math.nan
            )

            pred_perf_p50.append(
                quantile(by_turn_pred_perf[t], 0.50)
                if by_turn_pred_perf[t]
                else math.nan
            )
            correct_mean.append(
                (sum(by_turn_correct[t]) / float(len(by_turn_correct[t])))
                if by_turn_correct[t]
                else math.nan
            )

            pred_w_p50.append(
                quantile(by_turn_pred_welfare[t], 0.50)
                if by_turn_pred_welfare[t]
                else math.nan
            )
            obs_w_p50.append(
                quantile(by_turn_obs_welfare[t], 0.50)
                if by_turn_obs_welfare[t]
                else math.nan
            )

        plt.figure(figsize=(12, 16))

        ax1 = plt.subplot(5, 1, 1)
        ax1.plot(t_x, obs_cache_p50, label="obs_cache_ratio p50", linewidth=1.8)
        ax1.fill_between(
            t_x,
            obs_cache_p25,
            obs_cache_p75,
            alpha=0.2,
            label="obs_cache_ratio p25-p75",
        )
        ax1.plot(
            t_x, pred_cache_p50, label="pred_cache_ratio p50", linewidth=1.2, alpha=0.9
        )
        ax1.fill_between(
            t_x,
            pred_cache_p25,
            pred_cache_p75,
            alpha=0.15,
            label="pred_cache_ratio p25-p75",
        )
        ax1.set_title("Per-turn profile across dialogues (canonical per-turn)")
        ax1.set_xlabel("turn_number")
        ax1.set_ylabel("ratio")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="best")

        ax2 = plt.subplot(5, 1, 2, sharex=ax1)
        ax2.plot(t_x, lat_p50, label="obs_latency_ms p50", linewidth=1.8)
        ax2.fill_between(
            t_x, lat_p25, lat_p75, alpha=0.2, label="obs_latency_ms p25-p75"
        )
        ax2.set_xlabel("turn_number")
        ax2.set_ylabel("latency (ms)")
        ax2.legend(loc="best")

        ax3 = plt.subplot(5, 1, 3, sharex=ax1)
        ax3.plot(t_x, obs_cost_p50, label="obs_cost_tokens p50", linewidth=1.8)
        ax3.plot(
            t_x, pred_cost_p50, label="pred_cost_tokens p50", linewidth=1.2, alpha=0.9
        )
        ax3.set_xlabel("turn_number")
        ax3.set_ylabel("cost proxy (units)")
        ax3.legend(loc="best")

        ax4 = plt.subplot(5, 1, 4, sharex=ax1)
        ax4.plot(t_x, obs_w_p50, label="obs_welfare p50", linewidth=1.8)
        ax4.plot(t_x, pred_w_p50, label="pred_welfare p50", linewidth=1.2, alpha=0.9)
        ax4.set_xlabel("turn_number")
        ax4.set_ylabel("welfare (units)")
        ax4.legend(loc="best")

        ax5 = plt.subplot(5, 1, 5, sharex=ax1)
        ax5.plot(t_x, pred_perf_p50, label="pred_perf_prob p50", linewidth=1.5)
        ax5.plot(t_x, correct_mean, label="mean correct", linewidth=1.5, alpha=0.85)
        ax5.set_xlabel("turn_number")
        ax5.set_ylabel("prob / rate")
        ax5.set_ylim(-0.05, 1.05)
        ax5.legend(loc="best")

        plt.tight_layout()
        plt.savefig(outdir / "per_turn_profiles.png", dpi=160)
        plt.close()

    # Per-dialogue trace plots (includes welfare + payments)
    must_include: set[str] = set()
    for r in top_lat:
        must_include.add(_s(r.get("dialogue_id")))

    candidates = [
        s for s in dialogue_series if len(s.turns) >= int(max(1, min_dialogue_turns))
    ]
    candidates_sorted = sorted(
        candidates, key=lambda s: (len(s.turns), max(s.turns)), reverse=True
    )

    selected: list[DialogueSeries] = []
    did_to_series: dict[str, DialogueSeries] = {
        s.dialogue_id: s for s in candidates_sorted
    }

    for did in sorted(must_include):
        s = did_to_series.get(did)
        if s is not None:
            selected.append(s)

    selected_ids = {x.dialogue_id for x in selected}
    for s in candidates_sorted:
        if len(selected) >= int(max(0, max_dialogue_plots)):
            break
        if s.dialogue_id in selected_ids:
            continue
        selected.append(s)
        selected_ids.add(s.dialogue_id)

    def _plot_dialogue_trace(s: DialogueSeries, out_path: Path) -> None:
        xs: list[int] = s.turns

        # map backend_id strings to small integers for plotting
        backend_order: list[str] = []
        backend_to_idx: dict[str, int] = {}
        backend_idx: list[int] = []
        for b in s.backend_id:
            if b not in backend_to_idx:
                backend_to_idx[b] = len(backend_order)
                backend_order.append(b)
            backend_idx.append(backend_to_idx[b])

        corr_cl = _pearsonr_finite(s.obs_cache_ratio, s.obs_latency_ms)

        obs_lat_f: list[float] = [x for x in s.obs_latency_ms if _is_finite(float(x))]
        mean_lat = float(sum(obs_lat_f) / len(obs_lat_f)) if obs_lat_f else math.nan
        p90_lat = quantile(obs_lat_f, 0.90) if obs_lat_f else math.nan

        sid = s.dialogue_id if len(s.dialogue_id) <= 24 else s.dialogue_id[:24]
        backend_ids_joined = ",".join(sorted(set(s.backend_id)))

        plt.figure(figsize=(12, 22))

        ax0 = plt.subplot(7, 1, 1)
        ax0.step(xs, backend_idx, where="mid", linewidth=1.5)
        ax0.set_ylabel("backend")
        if backend_order:
            ax0.set_yticks(list(range(len(backend_order))))
            ax0.set_yticklabels(backend_order)
        ax0.grid(True, alpha=0.25)

        ax1 = plt.subplot(7, 1, 2, sharex=ax0)
        ax1.plot(
            xs, s.obs_cache_ratio, marker="o", label="obs_cache_ratio", linewidth=1.8
        )
        ax1.plot(
            xs,
            s.pred_cache_ratio,
            marker="o",
            label="pred_cache_ratio",
            linewidth=1.2,
            alpha=0.85,
        )
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("ratio")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.25)

        ax2 = plt.subplot(7, 1, 3, sharex=ax0)
        ax2.plot(
            xs, s.obs_latency_ms, marker="o", label="obs_latency_ms", linewidth=1.8
        )
        ax2.plot(
            xs,
            s.pred_latency_ms,
            marker="o",
            label="pred_latency_ms",
            linewidth=1.2,
            alpha=0.85,
        )
        ax2.set_ylabel("ms")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.25)

        ax3 = plt.subplot(7, 1, 4, sharex=ax0)
        ax3.plot(
            xs,
            s.obs_prompt_tokens,
            marker="o",
            label="obs_prompt_tokens",
            linewidth=1.5,
        )
        ax3.plot(
            xs,
            s.obs_cached_tokens,
            marker="o",
            label="obs_cached_tokens",
            linewidth=1.5,
        )
        ax3.set_ylabel("tokens")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.25)

        ax4 = plt.subplot(7, 1, 5, sharex=ax0)
        ax4.plot(
            xs, s.obs_cost_tokens, marker="o", label="obs_cost_tokens", linewidth=1.8
        )
        ax4.plot(
            xs,
            s.pred_cost_tokens,
            marker="o",
            label="pred_cost_tokens",
            linewidth=1.2,
            alpha=0.85,
        )
        ax4.set_ylabel("cost proxy")
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.25)

        ax5 = plt.subplot(7, 1, 6, sharex=ax0)
        ax5.plot(xs, s.obs_welfare, marker="o", label="obs_welfare", linewidth=1.8)
        ax5.plot(
            xs,
            s.pred_welfare,
            marker="o",
            label="pred_welfare",
            linewidth=1.2,
            alpha=0.85,
        )
        ax5.plot(
            xs,
            s.vcg_total_payment,
            marker="o",
            label="vcg_total_payment",
            linewidth=1.0,
            alpha=0.75,
        )
        ax5.set_ylabel("welfare / payment")
        ax5.legend(loc="best")
        ax5.grid(True, alpha=0.25)

        ax6 = plt.subplot(7, 1, 7, sharex=ax0)
        correct_float: list[float] = [1.0 if c else 0.0 for c in s.correct]
        ax6.plot(
            xs, s.pred_perf_prob, marker="o", label="pred_perf_prob", linewidth=1.5
        )
        ax6.plot(
            xs,
            correct_float,
            marker="o",
            label="correct (0/1)",
            linewidth=1.0,
            alpha=0.7,
        )
        ax6.set_xlabel("turn_number")
        ax6.set_ylabel("prob / label")
        ax6.set_ylim(-0.05, 1.05)
        ax6.legend(loc="best")
        ax6.grid(True, alpha=0.25)

        title = (
            f"Dialogue trace did={sid} turns={len(xs)} backends={backend_ids_joined} "
            f"mean_lat={mean_lat:.1f}ms p90_lat={p90_lat:.1f}ms "
            f"corr(cache,lat)={corr_cl:.3f}"
        )
        plt.suptitle(title, y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.975))
        plt.savefig(out_path, dpi=160)
        plt.close()

    for s in selected:
        stem = _sanitize_file_stem(s.dialogue_id)
        out_path = dialogues_dir / f"dialogue_trace__{stem}.png"
        _plot_dialogue_trace(s, out_path)

    print(f"\nWrote plots to: {outdir.resolve()}")
    if selected:
        print(
            f"Wrote per-dialogue traces to: {dialogues_dir.resolve()} (n={len(selected)})"
        )
