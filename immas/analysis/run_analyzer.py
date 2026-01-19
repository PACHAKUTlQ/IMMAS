"""
immas.analysis.run_analyzer

Analyze a JSONL router run log.

Outputs
-------
- Summary metrics (MAE/RMSE/correlation/R^2) for latency + cost + cache ratio
- Cache/KV diagnostics:
  - kvmatch_text proxy vs observed obs_cache_ratio
  - predicted cache ratio vs observed cache ratio (scatter + reliability curve)
  - binned latency vs cache_ratio (to visualize cache benefit)
- Outliers:
  - Top observed latency
  - Top latency residual (obs - pred)
  - Top cache-ratio residual (obs - pred)
  - Suspicious cases (kvmatch_text vs obs_cache_ratio mismatch)
  - Inconsistent usage cases (cached_tokens > prompt_tokens)
- Conversation-ordered CSV (turns_sorted.csv) for debugging KV cache behavior per dialogue
- Dialogue summary CSV (dialogue_summary.csv)
- Plots (if matplotlib is available):
  - Completion-order figures (model evolution over time) (kept)
  - Within-dialogue visualizations aligned by turn_number
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

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
    _short_id,
    _write_turns_csv,
)


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


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _finite_pairs(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """
    Filter (x, y) pairs where both are finite, preserving alignment.
    """
    out_x: List[float] = []
    out_y: List[float] = []
    for x, y in zip(xs, ys):
        xf = float(x)
        yf = float(y)
        if _is_finite(xf) and _is_finite(yf):
            out_x.append(xf)
            out_y.append(yf)
    return out_x, out_y


def _pearsonr_finite(xs: Sequence[float], ys: Sequence[float]) -> float:
    x2, y2 = _finite_pairs(xs, ys)
    return pearsonr(x2, y2)


def _safe_float_series(records: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    """
    Extract a float series from records. Missing/unparseable values become NaN.

    This is intended for plotting (where NaNs are acceptable).
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
        obs_latency_ms=[_f(r.get("obs_latency_ms"), math.nan) for r in per_turn],
        pred_latency_ms=[_f(r.get("pred_latency_ms"), math.nan) for r in per_turn],
        obs_cache_ratio=[_f(r.get("obs_cache_ratio"), math.nan) for r in per_turn],
        pred_cache_ratio=[_f(r.get("pred_cache_ratio"), math.nan) for r in per_turn],
        kvmatch_text=[_f(r.get("kvmatch_text"), math.nan) for r in per_turn],
        obs_prompt_tokens=[_i(r.get("obs_prompt_tokens")) for r in per_turn],
        obs_cached_tokens=[_i(r.get("obs_cached_tokens")) for r in per_turn],
        prompt_chars=[_i(r.get("prompt_chars")) for r in per_turn],
        cached_prompt_chars=[_i(r.get("cached_prompt_chars")) for r in per_turn],
        kvmatch_lcp_chars=[_i(r.get("kvmatch_lcp_chars")) for r in per_turn],
    )


def _binned_means(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_bins: int = 20,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute binned means of y over x in [x_min, x_max].

    Returns
    -------
    centers, means, counts
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    pairs: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = float(x)
        yf = float(y)
        if not (_is_finite(xf) and _is_finite(yf)):
            continue
        xc = max(x_min, min(x_max, xf))
        pairs.append((xc, yf))

    if not pairs:
        return [], [], []

    width = (x_max - x_min) / float(n_bins)
    if width <= 0:
        return [], [], []

    sums = [0.0] * n_bins
    counts = [0] * n_bins
    for x, y in pairs:
        idx = int((x - x_min) / width)
        idx = min(n_bins - 1, max(0, idx))
        sums[idx] += y
        counts[idx] += 1

    centers: List[float] = []
    means_out: List[float] = []
    counts_out: List[int] = []
    for i in range(n_bins):
        c = counts[i]
        if c <= 0:
            continue
        centers.append(x_min + (i + 0.5) * width)
        means_out.append(sums[i] / float(c))
        counts_out.append(c)

    return centers, means_out, counts_out


def _write_dialogue_summary_csv(
    *,
    out_path: Path,
    series: Sequence[DialogueSeries],
) -> None:
    """
    Write per-dialogue summary CSV for quick debugging/sorting.
    """
    cols = [
        "dialogue_id",
        "n_turns",
        "max_turn",
        "mean_obs_latency_ms",
        "p90_obs_latency_ms",
        "mean_obs_cache_ratio",
        "mean_pred_cache_ratio",
        "mean_kvmatch_text",
        "corr_cache_vs_latency",
        "corr_kvmatch_vs_cache",
        "mean_obs_prompt_tokens",
        "mean_obs_cached_tokens",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for s in series:
            obs_lat = [x for x in s.obs_latency_ms if _is_finite(x)]
            obs_cr = [x for x in s.obs_cache_ratio if _is_finite(x)]
            pred_cr = [x for x in s.pred_cache_ratio if _is_finite(x)]
            kv = [x for x in s.kvmatch_text if _is_finite(x)]

            corr_cache_lat = _pearsonr_finite(s.obs_cache_ratio, s.obs_latency_ms)
            corr_kv_cache = _pearsonr_finite(s.kvmatch_text, s.obs_cache_ratio)

            row: Dict[str, Any] = {
                "dialogue_id": s.dialogue_id,
                "n_turns": len(s.turns),
                "max_turn": max(s.turns) if s.turns else 0,
                "mean_obs_latency_ms": mean(obs_lat),
                "p90_obs_latency_ms": quantile(obs_lat, 0.90),
                "mean_obs_cache_ratio": mean(obs_cr),
                "mean_pred_cache_ratio": mean(pred_cr),
                "mean_kvmatch_text": mean(kv),
                "corr_cache_vs_latency": corr_cache_lat,
                "corr_kvmatch_vs_cache": corr_kv_cache,
                "mean_obs_prompt_tokens": mean([float(x) for x in s.obs_prompt_tokens]),
                "mean_obs_cached_tokens": mean([float(x) for x in s.obs_cached_tokens]),
            }
            w.writerow(row)


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

    # Pairwise metrics (skips missing/unparseable)
    pred_lat, obs_lat = _pairs(ok_by_end, "pred_latency_ms", "obs_latency_ms")
    pred_cost, obs_cost = _pairs(ok_by_end, "pred_cost_tokens", "obs_total_tokens")
    pred_cache, obs_cache = _pairs(ok_by_end, "pred_cache_ratio", "obs_cache_ratio")
    kvmatch, obs_cache2 = _pairs(ok_by_end, "kvmatch_text", "obs_cache_ratio")

    # Plot-friendly series (NaN for missing)
    obs_cache_ratio_all = _safe_float_series(ok_by_end, "obs_cache_ratio")
    pred_cache_ratio_all = _safe_float_series(ok_by_end, "pred_cache_ratio")
    kvmatch_all = _safe_float_series(ok_by_end, "kvmatch_text")

    obs_prompt_tokens_all = _safe_int_series(ok_by_end, "obs_prompt_tokens")
    obs_cached_tokens_all = _safe_int_series(ok_by_end, "obs_cached_tokens")

    correct = [bool(r.get("correct", True)) for r in ok_by_end]
    acc = sum(1 for c in correct if c) / len(correct) if correct else 0.0

    obs_lat_finite = [x for x in obs_lat if _is_finite(x)]
    obs_cost_finite = [x for x in obs_cost if _is_finite(x)]

    print("\nPrediction accuracy / regression metrics")
    print("---------------------------------------")
    print(f"Accuracy(correct):    {acc:.4f}")

    print(f"Latency pairs:        n={len(obs_lat)}")
    print(f"Latency MAE (ms):     {mae(pred_lat, obs_lat):.2f}")
    print(f"Latency RMSE (ms):    {rmse(pred_lat, obs_lat):.2f}")
    print(f"Latency corr:         {pearsonr(pred_lat, obs_lat):.3f}")
    print(f"Latency R^2:          {r2_score(pred_lat, obs_lat):.3f}")
    if obs_lat_finite:
        print(
            "Latency quantiles (ms): "
            f"p50={quantile(obs_lat_finite, 0.50):.1f}  "
            f"p90={quantile(obs_lat_finite, 0.90):.1f}  "
            f"p99={quantile(obs_lat_finite, 0.99):.1f}"
        )

    print(f"\nCost pairs:           n={len(obs_cost)}")
    print(f"Cost MAE (tok):       {mae(pred_cost, obs_cost):.2f}")
    print(f"Cost RMSE (tok):      {rmse(pred_cost, obs_cost):.2f}")
    print(f"Cost corr:            {pearsonr(pred_cost, obs_cost):.3f}")
    print(f"Cost R^2:             {r2_score(pred_cost, obs_cost):.3f}")
    if obs_cost_finite:
        print(
            "Cost quantiles (tok): "
            f"p50={quantile(obs_cost_finite, 0.50):.1f}  "
            f"p90={quantile(obs_cost_finite, 0.90):.1f}  "
            f"p99={quantile(obs_cost_finite, 0.99):.1f}"
        )

    print("\nKV cache / prefix reuse")
    print("-----------------------")
    print(
        f"Mean obs_prompt_tokens:  {mean([float(x) for x in obs_prompt_tokens_all]):.1f}"
    )
    print(
        f"Mean obs_cached_tokens:  {mean([float(x) for x in obs_cached_tokens_all]):.1f}"
    )

    obs_cr_finite = [x for x in obs_cache_ratio_all if _is_finite(x)]
    if obs_cr_finite:
        print(f"Mean obs_cache_ratio:    {mean(obs_cr_finite):.3f}")
        print(
            "obs_cache_ratio quantiles: "
            f"p50={quantile(obs_cr_finite, 0.50):.3f}  "
            f"p90={quantile(obs_cr_finite, 0.90):.3f}  "
            f"p99={quantile(obs_cr_finite, 0.99):.3f}"
        )
    else:
        print("Mean obs_cache_ratio:    0.000")

    if pred_cache and obs_cache:
        print(f"CacheRatio pairs:        n={len(obs_cache)}")
        print(f"CacheRatio MAE:          {mae(pred_cache, obs_cache):.3f}")
        print(f"CacheRatio RMSE:         {rmse(pred_cache, obs_cache):.3f}")
        print(f"CacheRatio corr:         {pearsonr(pred_cache, obs_cache):.3f}")
        print(f"CacheRatio R^2:          {r2_score(pred_cache, obs_cache):.3f}")
    if kvmatch and obs_cache2:
        print(f"kvmatch_text corr(obs):  {pearsonr(kvmatch, obs_cache2):.3f}")

    obs_lat_all_plot = _safe_float_series(ok_by_end, "obs_latency_ms")
    cache_lat_corr = _pearsonr_finite(obs_cache_ratio_all, obs_lat_all_plot)
    print(f"\nObs corr(cache_ratio, latency): {cache_lat_corr:.3f}")

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

    # Check turn continuity per dialogue_id
    gaps: List[Tuple[str, List[int]]] = []
    for did, rs in by_did.items():
        turns = sorted(
            {_i(r.get("turn_number")) for r in rs if _i(r.get("turn_number")) > 0}
        )
        if not turns:
            continue
        expected = list(range(turns[0], turns[-1] + 1))
        if turns != expected:
            gaps.append((did, turns))

    if gaps:
        print(
            "\nWARNING: turn-number gaps/duplicates detected (may indicate retries or partial runs)"
        )
        print(
            "--------------------------------------------------------------------------"
        )
        for did, turns in gaps[:10]:
            print(f"did={_short_id(did)} turns={turns}")
        if len(gaps) > 10:
            print(f"... {len(gaps) - 10} more")
    else:
        print("\nTurn-number continuity: OK (per dialogue_id)")

    # Latency outliers (observed)
    topk = int(max(1, args.topk))
    top_lat = sorted(
        ok_by_end, key=lambda r: _f(r.get("obs_latency_ms")), reverse=True
    )[:topk]
    print("\nTop latency outliers (with cache context)")
    print("----------------------------------------")
    for r in top_lat:
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        obs_ms = _f(r.get("obs_latency_ms"))
        pred_ms = _f(r.get("pred_latency_ms"))
        kv = _f(r.get("kvmatch_text"))
        pr = _f(r.get("pred_cache_ratio"))
        ocr = _f(r.get("obs_cache_ratio"))
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        inflight = _i(r.get("router_inflight"))
        rps = _f(r.get("router_rps_1s"))
        print(
            f"did={did} turn={turn} obs_ms={obs_ms:.1f} pred_ms={pred_ms:.1f} "
            f"kvmatch={kv:.3f} pred_cr={pr:.3f} obs_cr={ocr:.3f} "
            f"prompt_tok={pt} cached_tok={ct} inflight={inflight} rps_1s={rps:.1f}"
        )

    # Residual outliers (obs - pred)
    lat_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    cache_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    for r in ok_by_end:
        ol = _f(r.get("obs_latency_ms"), math.nan)
        pl = _f(r.get("pred_latency_ms"), math.nan)
        if _is_finite(ol) and _is_finite(pl):
            lat_residuals.append((ol - pl, r))

        oc = _f(r.get("obs_cache_ratio"), math.nan)
        pc = _f(r.get("pred_cache_ratio"), math.nan)
        if _is_finite(oc) and _is_finite(pc):
            cache_residuals.append((oc - pc, r))

    lat_residuals_sorted = sorted(lat_residuals, key=lambda x: abs(x[0]), reverse=True)[
        :topk
    ]
    cache_residuals_sorted = sorted(
        cache_residuals, key=lambda x: abs(x[0]), reverse=True
    )[:topk]

    if lat_residuals_sorted:
        print("\nTop |latency residual| outliers (obs - pred, with cache context)")
        print("---------------------------------------------------------------")
        for resid, r in lat_residuals_sorted:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            obs_ms = _f(r.get("obs_latency_ms"))
            pred_ms = _f(r.get("pred_latency_ms"))
            kv = _f(r.get("kvmatch_text"))
            ocr = _f(r.get("obs_cache_ratio"))
            pt = _i(r.get("obs_prompt_tokens"))
            ct = _i(r.get("obs_cached_tokens"))
            print(
                f"did={did} turn={turn} resid_ms={resid:.1f} obs_ms={obs_ms:.1f} pred_ms={pred_ms:.1f} "
                f"obs_cr={ocr:.3f} kvmatch={kv:.3f} prompt_tok={pt} cached_tok={ct}"
            )

    if cache_residuals_sorted:
        print("\nTop |cache_ratio residual| outliers (obs - pred)")
        print("------------------------------------------------")
        for resid, r in cache_residuals_sorted:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            oc = _f(r.get("obs_cache_ratio"))
            pc = _f(r.get("pred_cache_ratio"))
            kv = _f(r.get("kvmatch_text"))
            lcp = _i(r.get("kvmatch_lcp_chars"))
            print(
                f"did={did} turn={turn} resid_cr={resid:+.3f} obs_cr={oc:.3f} pred_cr={pc:.3f} "
                f"kvmatch={kv:.3f} lcp_chars={lcp}"
            )

    # Suspicious KV cases: large mismatch between text proxy and observed cache ratio
    suspicious: List[Mapping[str, Any]] = []
    for r in ok_by_end:
        kv = _f(r.get("kvmatch_text"), math.nan)
        ocr = _f(r.get("obs_cache_ratio"), math.nan)
        if not (_is_finite(kv) and _is_finite(ocr)):
            continue
        if (
            abs(kv - ocr) >= 0.7
            or (kv >= 0.9 and ocr <= 0.1)
            or (kv <= 0.1 and ocr >= 0.9)
        ):
            suspicious.append(r)

    if suspicious:
        print("\nSuspicious KV cases (kvmatch_text vs obs_cache_ratio mismatch)")
        print("------------------------------------------------------------")
        for r in suspicious[:topk]:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            kv = _f(r.get("kvmatch_text"))
            ocr = _f(r.get("obs_cache_ratio"))
            pt = _i(r.get("obs_prompt_tokens"))
            ct = _i(r.get("obs_cached_tokens"))
            lcp = _i(r.get("kvmatch_lcp_chars"))
            cached_chars = _i(r.get("cached_prompt_chars"))
            prompt_chars = _i(r.get("prompt_chars"))
            print(
                f"did={did} turn={turn} kvmatch={kv:.3f} obs_cr={ocr:.3f} "
                f"prompt_tok={pt} cached_tok={ct} "
                f"lcp_chars={lcp} cached_chars={cached_chars} prompt_chars={prompt_chars}"
            )

    # Inconsistent usage cases: cached_tokens > prompt_tokens
    inconsistent: List[Mapping[str, Any]] = []
    for r in ok_by_end:
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        if pt > 0 and ct > pt:
            inconsistent.append(r)

    if inconsistent:
        print("\nWARNING: inconsistent usage (obs_cached_tokens > obs_prompt_tokens)")
        print("---------------------------------------------------------------")
        for r in inconsistent[:topk]:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            pt = _i(r.get("obs_prompt_tokens"))
            ct = _i(r.get("obs_cached_tokens"))
            ocr = _f(r.get("obs_cache_ratio"))
            print(
                f"did={did} turn={turn} prompt_tok={pt} cached_tok={ct} obs_cr={ocr:.3f}"
            )

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

    # Aggregate per-turn stats across dialogues (canonical per-turn)
    by_turn_cache: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_kvmatch: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_latency: DefaultDict[int, List[float]] = defaultdict(list)
    by_turn_prompt_tok: DefaultDict[int, List[int]] = defaultdict(list)
    by_turn_cached_tok: DefaultDict[int, List[int]] = defaultdict(list)

    for s in dialogue_series:
        for t, cr, kv, lat, pt, ct in zip(
            s.turns,
            s.obs_cache_ratio,
            s.kvmatch_text,
            s.obs_latency_ms,
            s.obs_prompt_tokens,
            s.obs_cached_tokens,
        ):
            if _is_finite(cr):
                by_turn_cache[t].append(float(cr))
            if _is_finite(kv):
                by_turn_kvmatch[t].append(float(kv))
            if _is_finite(lat):
                by_turn_latency[t].append(float(lat))
            by_turn_prompt_tok[t].append(int(pt))
            by_turn_cached_tok[t].append(int(ct))

    # Print a compact per-turn table
    turns_sorted = sorted(by_turn_latency.keys())
    if turns_sorted:
        print("\nPer-turn aggregates (canonical: last record per turn per dialogue)")
        print("---------------------------------------------------------------")
        print(
            "turn  n   mean_lat(ms)  p90_lat  mean_cache  mean_kvmatch  mean_prompt_tok  mean_cached_tok"
        )
        for t in turns_sorted[:30]:
            lats = by_turn_latency[t]
            crs = by_turn_cache[t]
            kvs = by_turn_kvmatch[t]
            pts = by_turn_prompt_tok[t]
            cts = by_turn_cached_tok[t]
            print(
                f"{t:>4}  {len(lats):>3}  "
                f"{mean(lats):>11.1f}  {quantile(lats, 0.90):>7.1f}  "
                f"{mean(crs):>10.3f}  {mean(kvs):>12.3f}  "
                f"{mean([float(x) for x in pts]):>15.1f}  {mean([float(x) for x in cts]):>15.1f}"
            )
        if len(turns_sorted) > 30:
            print(f"... ({len(turns_sorted) - 30} more turns)")

    # Plots (optional)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available; skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = outdir / "dialogues"
    dialogues_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # completion-order evolution plots
    # -------------------------------------------------------------------------
    xs_all = list(range(len(ok_by_end)))
    pred_lat_all_plot = _safe_float_series(ok_by_end, "pred_latency_ms")

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

    # Time series: cost (new; uses previously-unused series)
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

    # Time series: cache ratio & kvmatch
    plt.figure(figsize=(12, 5))
    plt.plot(xs_all, obs_cache_ratio_all, label="observed cache ratio", linewidth=1.5)
    plt.plot(
        xs_all,
        pred_cache_ratio_all,
        label="predicted cache ratio",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.plot(
        xs_all, kvmatch_all, label="kvmatch_text (proxy)", linewidth=1.0, alpha=0.8
    )
    plt.title("KV cache reuse over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("ratio")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cache_ratio_timeseries.png", dpi=160)
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

    # Scatter: predicted vs observed cost (new)
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

    # Scatter: kvmatch_text vs observed cache ratio
    if kvmatch and obs_cache2:
        plt.figure(figsize=(6, 6))
        plt.scatter(kvmatch, obs_cache2, s=10, alpha=0.6)
        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.5,
        )
        plt.title("kvmatch_text (proxy) vs observed cache ratio")
        plt.xlabel("kvmatch_text")
        plt.ylabel("obs_cache_ratio")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(outdir / "kvmatch_vs_obs_cache_ratio.png", dpi=160)
        plt.close()

    # Scatter: observed cache ratio vs observed latency
    xs_cr_lat, ys_cr_lat = _finite_pairs(obs_cache_ratio_all, obs_lat_all_plot)
    if xs_cr_lat and ys_cr_lat:
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_cr_lat, ys_cr_lat, s=10, alpha=0.6)
        plt.title("Observed cache ratio vs observed latency")
        plt.xlabel("obs_cache_ratio")
        plt.ylabel("obs_latency_ms")
        plt.xlim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(outdir / "obs_cache_ratio_vs_latency.png", dpi=160)
        plt.close()

    # -------------------------------------------------------------------------
    # Additional figures: distributions, residuals, calibration
    # -------------------------------------------------------------------------
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

    if pred_cache and obs_cache:
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_cache, obs_cache, s=10, alpha=0.6)
        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.5,
        )
        plt.title("Predicted vs observed cache ratio")
        plt.xlabel("pred_cache_ratio")
        plt.ylabel("obs_cache_ratio")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(outdir / "pred_cache_ratio_vs_obs_cache_ratio.png", dpi=160)
        plt.close()

        centers, means_y, _counts = _binned_means(
            pred_cache, obs_cache, n_bins=20, x_min=0.0, x_max=1.0
        )
        if centers and means_y:
            plt.figure(figsize=(7, 5))
            plt.plot(
                centers,
                means_y,
                marker="o",
                linewidth=1.5,
                label="mean obs_cache_ratio per pred bin",
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
            plt.title("Cache ratio calibration (reliability curve)")
            plt.xlabel("pred_cache_ratio (binned)")
            plt.ylabel("mean obs_cache_ratio")
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "cache_ratio_calibration_curve.png", dpi=160)
            plt.close()

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

    if pred_cache and obs_cache:
        resid = [o - p for p, o in zip(pred_cache, obs_cache)]
        plt.figure(figsize=(7, 5))
        plt.hist(resid, bins=60, alpha=0.85)
        plt.title("Cache-ratio residuals distribution (obs - pred)")
        plt.xlabel("residual_cache_ratio")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "cache_ratio_residuals_hist.png", dpi=160)
        plt.close()

    # Cached tokens vs prompt tokens (sanity scatter)
    if obs_prompt_tokens_all and obs_cached_tokens_all:
        xs_pt: List[float] = []
        ys_ct: List[float] = []
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
    # Within-dialogue / per-turn visualization
    # -------------------------------------------------------------------------
    turns = sorted(by_turn_latency.keys())
    if turns:

        def _band(vals: Sequence[float]) -> Tuple[float, float, float]:
            v = [x for x in vals if _is_finite(float(x))]
            if not v:
                return (math.nan, math.nan, math.nan)
            return (quantile(v, 0.25), quantile(v, 0.50), quantile(v, 0.75))

        t_x: List[int] = []
        cache_p25: List[float] = []
        cache_p50: List[float] = []
        cache_p75: List[float] = []

        kv_p25: List[float] = []
        kv_p50: List[float] = []
        kv_p75: List[float] = []

        lat_p25: List[float] = []
        lat_p50: List[float] = []
        lat_p75: List[float] = []

        for t in turns:
            t_x.append(t)

            a, b, c = _band(by_turn_cache[t])
            cache_p25.append(a)
            cache_p50.append(b)
            cache_p75.append(c)

            a, b, c = _band(by_turn_kvmatch[t])
            kv_p25.append(a)
            kv_p50.append(b)
            kv_p75.append(c)

            a, b, c = _band(by_turn_latency[t])
            lat_p25.append(a)
            lat_p50.append(b)
            lat_p75.append(c)

        plt.figure(figsize=(12, 10))

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t_x, cache_p50, label="obs_cache_ratio p50", linewidth=1.8)
        ax1.fill_between(
            t_x, cache_p25, cache_p75, alpha=0.2, label="obs_cache_ratio p25-p75"
        )
        ax1.plot(t_x, kv_p50, label="kvmatch_text p50", linewidth=1.2, alpha=0.9)
        ax1.fill_between(t_x, kv_p25, kv_p75, alpha=0.15, label="kvmatch_text p25-p75")
        ax1.set_title("Per-turn profile across dialogues (canonical per-turn)")
        ax1.set_xlabel("turn_number")
        ax1.set_ylabel("ratio")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="best")

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(t_x, lat_p50, label="obs_latency_ms p50", linewidth=1.8)
        ax2.fill_between(
            t_x, lat_p25, lat_p75, alpha=0.2, label="obs_latency_ms p25-p75"
        )
        ax2.set_xlabel("turn_number")
        ax2.set_ylabel("latency (ms)")
        ax2.legend(loc="best")

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        counts = [len(by_turn_latency[t]) for t in t_x]
        ax3.bar(t_x, counts, alpha=0.85)
        ax3.set_xlabel("turn_number")
        ax3.set_ylabel("n dialogues (with this turn)")
        plt.tight_layout()
        plt.savefig(outdir / "per_turn_profiles.png", dpi=160)
        plt.close()

    # Per-dialogue trace plots
    must_include: set[str] = set()
    for r in top_lat:
        must_include.add(_s(r.get("dialogue_id")))
    for r in suspicious[:topk]:
        must_include.add(_s(r.get("dialogue_id")))

    candidates = [
        s
        for s in dialogue_series
        if len(s.turns) >= int(max(1, args.min_dialogue_turns))
    ]
    candidates_sorted = sorted(
        candidates, key=lambda s: (len(s.turns), max(s.turns)), reverse=True
    )

    selected: List[DialogueSeries] = []
    did_to_series = {s.dialogue_id: s for s in candidates_sorted}
    for did in sorted(must_include):
        s = did_to_series.get(did)
        if s is not None:
            selected.append(s)

    for s in candidates_sorted:
        if len(selected) >= int(max(0, args.max_dialogue_plots)):
            break
        if s.dialogue_id in {x.dialogue_id for x in selected}:
            continue
        selected.append(s)

    def _plot_dialogue_trace(s: DialogueSeries, out_path: Path) -> None:
        xs = s.turns

        plt.figure(figsize=(12, 9))

        ax1 = plt.subplot(3, 1, 1)
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
        ax1.plot(
            xs,
            s.kvmatch_text,
            marker="o",
            label="kvmatch_text (proxy)",
            linewidth=1.2,
            alpha=0.85,
        )
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("ratio")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.25)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
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

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(
            xs, s.obs_latency_ms, marker="o", label="obs_latency_ms", linewidth=1.8
        )
        ax3.plot(
            xs,
            s.pred_latency_ms,
            marker="o",
            label="pred_latency_ms",
            linewidth=1.2,
            alpha=0.85,
        )
        ax3.set_xlabel("turn_number")
        ax3.set_ylabel("latency (ms)")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.25)

        sid = _short_id(s.dialogue_id, 24)
        corr_cl = _pearsonr_finite(s.obs_cache_ratio, s.obs_latency_ms)
        corr_kc = _pearsonr_finite(s.kvmatch_text, s.obs_cache_ratio)
        plt.suptitle(
            f"Dialogue trace did={sid}  turns={len(xs)}  "
            f"corr(cache,lat)={corr_cl:.3f}  corr(kvmatch,cache)={corr_kc:.3f}",
            y=0.99,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
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


if __name__ == "__main__":
    main()
