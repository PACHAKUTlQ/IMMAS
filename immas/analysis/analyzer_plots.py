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
from immas.analysis.utils import _s, _short_id, quantile, _is_finite, _pearsonr_finite


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
    correct_all = [bool(r.get("correct", True)) for r in ok_by_end]
    correct_all_float = [1.0 if c else 0.0 for c in correct_all]

    # Backend usage (categorical)
    backend_ids = [_s(r.get("backend_id")) or "backend_unknown" for r in ok_by_end]
    backend_counts: dict[str, int] = {}
    for bid in backend_ids:
        backend_counts[bid] = backend_counts.get(bid, 0) + 1

    obs_lat_finite = [x for x in obs_lat if _is_finite(x)]
    obs_cr_finite = [x for x in obs_cache_ratio_all if _is_finite(x)]

    # -------------------------------------------------------------------------
    # completion-order evolution plots
    # -------------------------------------------------------------------------
    xs_all = list(range(len(ok_by_end)))

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

    # Time series: cost
    obs_cost_all_plot = _safe_float_series(ok_by_end, "obs_total_tokens")
    pred_cost_all_plot = _safe_float_series(ok_by_end, "pred_cost_tokens")

    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_cost_all_plot, label="observed total tokens", linewidth=1.5)
    plt.plot(
        xs_all,
        pred_cost_all_plot,
        label="predicted cost (tokens)",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.title("Cost over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cost_timeseries.png", dpi=160)
    plt.close()

    # Time series: cache ratio (no mismatch deep-dive)
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

    # Scatter: predicted vs observed cost
    if pred_cost and obs_cost:
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_cost, obs_cost, s=10, alpha=0.6)
        lo = min(min(pred_cost), min(obs_cost))
        hi = max(max(pred_cost), max(obs_cost))
        plt.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5
        )
        plt.title("Predicted vs observed cost (tokens)")
        plt.xlabel("pred_cost_tokens")
        plt.ylabel("obs_total_tokens")
        plt.tight_layout()
        plt.savefig(outdir / "cost_scatter.png", dpi=160)
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

    if obs_cr_finite:
        plt.figure(figsize=(7, 5))
        plt.hist(obs_cr_finite, bins=40, range=(0.0, 1.0), alpha=0.85)
        plt.title("Observed cache ratio distribution")
        plt.xlabel("obs_cache_ratio")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "obs_cache_ratio_hist.png", dpi=160)
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

    # Residual histograms: latency, cost
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
        plt.xlabel("residual_tokens")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "cost_residuals_hist.png", dpi=160)
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

    # -------------------------------------------------------------------------
    # Within-dialogue / per-turn visualization (canonical per-turn)
    # -------------------------------------------------------------------------
    by_turn_obs_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_latency: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_cost: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_obs_total: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_pred_perf: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_correct: DefaultDict[int, List[float]] = defaultdict(list)

    for s in dialogue_series:
        for t, ocr, pcr, lat, pc, ot, pp, corr in zip(
            s.turns,
            s.obs_cache_ratio,
            s.pred_cache_ratio,
            s.obs_latency_ms,
            s.pred_cost_tokens,
            [float(x) for x in s.obs_total_tokens],
            s.pred_perf_prob,
            s.correct,
        ):
            if _is_finite(ocr):
                by_turn_obs_cache[t].append(float(ocr))
            if _is_finite(pcr):
                by_turn_pred_cache[t].append(float(pcr))
            if _is_finite(lat):
                by_turn_latency[t].append(float(lat))
            if _is_finite(pc):
                by_turn_pred_cost[t].append(float(pc))
            if _is_finite(ot):
                by_turn_obs_total[t].append(float(ot))
            if _is_finite(pp):
                by_turn_pred_perf[t].append(float(pp))
            by_turn_correct[t].append(1.0 if bool(corr) else 0.0)

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
        obs_total_p50: list[float] = []

        pred_perf_p50: list[float] = []
        correct_mean: list[float] = []

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
            obs_total_p50.append(
                quantile(by_turn_obs_total[t], 0.50)
                if by_turn_obs_total[t]
                else math.nan
            )

            pred_perf_p50.append(
                quantile(by_turn_pred_perf[t], 0.50)
                if by_turn_pred_perf[t]
                else math.nan
            )
            correct_mean.append(
                sum(by_turn_correct[t]) / float(len(by_turn_correct[t]))
                if by_turn_correct[t]
                else math.nan
            )

        plt.figure(figsize=(12, 13))

        ax1 = plt.subplot(4, 1, 1)
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

        ax2 = plt.subplot(4, 1, 2, sharex=ax1)
        ax2.plot(t_x, lat_p50, label="obs_latency_ms p50", linewidth=1.8)
        ax2.fill_between(
            t_x, lat_p25, lat_p75, alpha=0.2, label="obs_latency_ms p25-p75"
        )
        ax2.set_xlabel("turn_number")
        ax2.set_ylabel("latency (ms)")
        ax2.legend(loc="best")

        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t_x, obs_total_p50, label="obs_total_tokens p50", linewidth=1.8)
        ax3.plot(
            t_x, pred_cost_p50, label="pred_cost_tokens p50", linewidth=1.2, alpha=0.9
        )
        ax3.set_xlabel("turn_number")
        ax3.set_ylabel("tokens")
        ax3.legend(loc="best")

        ax4 = plt.subplot(4, 1, 4, sharex=ax1)
        ax4.plot(t_x, pred_perf_p50, label="pred_perf_prob p50", linewidth=1.5)
        ax4.plot(t_x, correct_mean, label="mean correct", linewidth=1.5, alpha=0.85)
        ax4.set_xlabel("turn_number")
        ax4.set_ylabel("prob / rate")
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(loc="best")

        plt.tight_layout()
        plt.savefig(outdir / "per_turn_profiles.png", dpi=160)
        plt.close()

    # Per-dialogue trace plots
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
    did_to_series = {s.dialogue_id: s for s in candidates_sorted}

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
        xs = s.turns

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
        sid = _short_id(s.dialogue_id, 24)
        backend_ids_joined = ",".join(sorted(set(s.backend_id)))

        plt.figure(figsize=(12, 15))

        ax0 = plt.subplot(5, 1, 1)
        ax0.step(xs, backend_idx, where="mid", linewidth=1.5)
        ax0.set_ylabel("backend")
        if backend_order:
            ax0.set_yticks(list(range(len(backend_order))))
            ax0.set_yticklabels(backend_order)
        ax0.grid(True, alpha=0.25)

        ax1 = plt.subplot(5, 1, 2, sharex=ax0)
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

        ax2 = plt.subplot(5, 1, 3, sharex=ax0)
        ax2.plot(
            xs,
            s.obs_prompt_tokens,
            marker="o",
            label="obs_prompt_tokens",
            linewidth=1.5,
        )
        ax2.plot(
            xs,
            s.obs_cached_tokens,
            marker="o",
            label="obs_cached_tokens",
            linewidth=1.5,
        )
        ax2.set_ylabel("tokens")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.25)

        ax3 = plt.subplot(5, 1, 4, sharex=ax0)
        ax3.plot(
            xs, s.obs_total_tokens, marker="o", label="obs_total_tokens", linewidth=1.8
        )
        ax3.plot(
            xs,
            s.pred_cost_tokens,
            marker="o",
            label="pred_cost_tokens",
            linewidth=1.2,
            alpha=0.85,
        )
        ax3.set_ylabel("tokens")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.25)

        ax4 = plt.subplot(5, 1, 5, sharex=ax0)
        correct_float = [1.0 if c else 0.0 for c in s.correct]
        ax4.plot(
            xs, s.pred_perf_prob, marker="o", label="pred_perf_prob", linewidth=1.5
        )
        ax4.plot(
            xs,
            correct_float,
            marker="o",
            label="correct (0/1)",
            linewidth=1.0,
            alpha=0.7,
        )
        ax4.set_xlabel("turn_number")
        ax4.set_ylabel("prob / label")
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.25)

        plt.suptitle(
            f"Dialogue trace did={sid} turns={len(xs)} backends={backend_ids_joined} "
            f"corr(cache,lat)={corr_cl:.3f}",
            y=0.995,
        )
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
            f"Wrote per-dialogue traces to: {dialogues_dir.resolve()} (n={
                len(selected)
            })"
        )
