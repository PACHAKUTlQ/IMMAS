"""
immas.analysis.analyzer_reports

CSV report writers for analyzer outputs.
"""

from __future__ import annotations

import csv
import math

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from immas.analysis.analyzer_metrics import (
    BinaryProbMetrics,
    compute_binary_prob_metrics,
    extract_perf_pairs_from_records,
)
from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import (
    _f,
    _s,
    mae,
    mean,
    quantile,
    _is_finite,
    _pearsonr_finite,
)


def _mode_string(xs: Sequence[str]) -> str:
    """
    Return the most common string in xs. Ties break by first-seen order.

    Returns empty string for empty input.
    """

    if not xs:
        return ""
    counts: dict[str, int] = {}
    order: list[str] = []
    for x in xs:
        if x not in counts:
            counts[x] = 0
            order.append(x)
        counts[x] += 1
    best = order[0]
    best_c = counts[best]
    for k in order[1:]:
        c = counts[k]
        if c > best_c:
            best = k
            best_c = c
    return best


def _count_switches(xs: Sequence[str]) -> int:
    """
    Count adjacent value changes in xs.
    """

    if len(xs) < 2:
        return 0
    sw = 0
    prev = xs[0]
    for x in xs[1:]:
        if x != prev:
            sw += 1
        prev = x
    return sw


def _write_dialogue_summary_csv(
    *,
    out_path: Path,
    series: Sequence[DialogueSeries],
) -> None:
    """
    Write per-dialogue summary CSV for quick debugging/sorting.

    Includes:
    - backend identity + switching
    - latency/cost prediction errors
    - cache reuse summary
    - performance-probability metrics (pred_perf_prob vs correct)
    """

    cols = [
        "dialogue_id",
        "n_turns",
        "max_turn",
        "backend_mode",
        "n_backends",
        "backend_switches",
        "backend_ids",
        "model_mode",
        "source_mode",
        "mean_obs_latency_ms",
        "p90_obs_latency_ms",
        "mean_pred_latency_ms",
        "latency_mae_ms",
        "mean_obs_total_tokens",
        "p90_obs_total_tokens",
        "mean_pred_cost_tokens",
        "cost_mae_tokens",
        "mean_obs_cache_ratio",
        "mean_pred_cache_ratio",
        "corr_cache_vs_latency",
        "mean_obs_prompt_tokens",
        "mean_obs_cached_tokens",
        "perf_n",
        "perf_mean_pred",
        "perf_mean_obs_acc",
        "perf_accuracy_at_0_5",
        "perf_brier",
        "perf_log_loss",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for s in series:
            obs_lat = [x for x in s.obs_latency_ms if _is_finite(x)]
            pred_lat = [x for x in s.pred_latency_ms if _is_finite(x)]
            obs_total = [float(x) for x in s.obs_total_tokens if _is_finite(float(x))]
            pred_cost = [x for x in s.pred_cost_tokens if _is_finite(x)]

            obs_cr = [x for x in s.obs_cache_ratio if _is_finite(x)]
            pred_cr = [x for x in s.pred_cache_ratio if _is_finite(x)]

            corr_cache_lat = _pearsonr_finite(s.obs_cache_ratio, s.obs_latency_ms)

            # Perf metrics per dialogue
            perf_pairs_p: list[float] = []
            perf_pairs_y: list[bool] = []
            for p, y in zip(s.pred_perf_prob, s.correct):
                pf = float(p)
                if _is_finite(pf):
                    perf_pairs_p.append(pf)
                    perf_pairs_y.append(bool(y))
            perf: BinaryProbMetrics = compute_binary_prob_metrics(
                perf_pairs_p, perf_pairs_y
            )

            backend_mode = _mode_string(s.backend_id)
            model_mode = _mode_string(s.model)
            source_mode = _mode_string(s.source)

            backend_ids_sorted = sorted({b for b in s.backend_id if b})
            backend_ids_joined = ",".join(backend_ids_sorted)

            row: Dict[str, Any] = {
                "dialogue_id": s.dialogue_id,
                "n_turns": len(s.turns),
                "max_turn": max(s.turns) if s.turns else 0,
                "backend_mode": backend_mode,
                "n_backends": len(set(s.backend_id)),
                "backend_switches": _count_switches(s.backend_id),
                "backend_ids": backend_ids_joined,
                "model_mode": model_mode,
                "source_mode": source_mode,
                "mean_obs_latency_ms": mean(obs_lat),
                "p90_obs_latency_ms": quantile(obs_lat, 0.90),
                "mean_pred_latency_ms": mean(pred_lat),
                "latency_mae_ms": mae(pred_lat, obs_lat),
                "mean_obs_total_tokens": mean(obs_total),
                "p90_obs_total_tokens": quantile(obs_total, 0.90),
                "mean_pred_cost_tokens": mean(pred_cost),
                "cost_mae_tokens": mae(pred_cost, obs_total),
                "mean_obs_cache_ratio": mean(obs_cr),
                "mean_pred_cache_ratio": mean(pred_cr),
                "corr_cache_vs_latency": corr_cache_lat,
                "mean_obs_prompt_tokens": mean([float(x) for x in s.obs_prompt_tokens]),
                "mean_obs_cached_tokens": mean([float(x) for x in s.obs_cached_tokens]),
                "perf_n": perf.n,
                "perf_mean_pred": perf.mean_pred,
                "perf_mean_obs_acc": perf.mean_obs,
                "perf_accuracy_at_0_5": perf.accuracy_at_0_5,
                "perf_brier": perf.brier,
                "perf_log_loss": perf.log_loss,
            }
            w.writerow(row)


def _write_backend_summary_csv(
    *,
    out_path: Path,
    ok_by_end: Sequence[Mapping[str, Any]],
) -> None:
    """
    Write per-backend summary CSV.

    Includes:
    - backend usage share
    - latency/cost regression errors
    - performance-probability metrics (pred_perf_prob vs correct)
    """

    cols = [
        "backend_id",
        "model",
        "backend_base_url_v1",
        "n_requests",
        "frac_requests",
        "mean_obs_latency_ms",
        "mean_pred_latency_ms",
        "latency_mae_ms",
        "mean_obs_total_tokens",
        "mean_pred_cost_tokens",
        "cost_mae_tokens",
        "mean_obs_cache_ratio",
        "perf_n",
        "perf_mean_pred",
        "perf_mean_obs_acc",
        "perf_accuracy_at_0_5",
        "perf_brier",
        "perf_log_loss",
    ]

    by_backend: dict[str, list[Mapping[str, Any]]] = {}
    for r in ok_by_end:
        bid = _s(r.get("backend_id")) or "backend_unknown"
        by_backend.setdefault(bid, []).append(r)

    total = sum(len(v) for v in by_backend.values())
    total_f = float(total) if total > 0 else 1.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for backend_id in sorted(by_backend.keys()):
            rs = by_backend[backend_id]
            n = len(rs)

            model = _s(rs[0].get("model")) if rs else ""
            base_url = _s(rs[0].get("backend_base_url_v1")) if rs else ""

            obs_lat: list[float] = []
            pred_lat: list[float] = []
            obs_cost: list[float] = []
            pred_cost: list[float] = []
            obs_cache: list[float] = []

            for r in rs:
                ol = _f(r.get("obs_latency_ms"), math.nan)
                pl = _f(r.get("pred_latency_ms"), math.nan)
                if _is_finite(ol) and _is_finite(pl):
                    obs_lat.append(ol)
                    pred_lat.append(pl)

                oc = _f(r.get("obs_total_tokens"), math.nan)
                pc = _f(r.get("pred_cost_tokens"), math.nan)
                if _is_finite(oc) and _is_finite(pc):
                    obs_cost.append(oc)
                    pred_cost.append(pc)

                ocr = _f(r.get("obs_cache_ratio"), math.nan)
                if _is_finite(ocr):
                    obs_cache.append(ocr)

            ps, ys = extract_perf_pairs_from_records([dict(r) for r in rs])
            perf: BinaryProbMetrics = compute_binary_prob_metrics(ps, ys)

            row: Dict[str, Any] = {
                "backend_id": backend_id,
                "model": model,
                "backend_base_url_v1": base_url,
                "n_requests": n,
                "frac_requests": float(n) / total_f,
                "mean_obs_latency_ms": mean(obs_lat),
                "mean_pred_latency_ms": mean(pred_lat),
                "latency_mae_ms": mae(pred_lat, obs_lat),
                "mean_obs_total_tokens": mean(obs_cost),
                "mean_pred_cost_tokens": mean(pred_cost),
                "cost_mae_tokens": mae(pred_cost, obs_cost),
                "mean_obs_cache_ratio": mean(obs_cache),
                "perf_n": perf.n,
                "perf_mean_pred": perf.mean_pred,
                "perf_mean_obs_acc": perf.mean_obs,
                "perf_accuracy_at_0_5": perf.accuracy_at_0_5,
                "perf_brier": perf.brier,
                "perf_log_loss": perf.log_loss,
            }
            w.writerow(row)
