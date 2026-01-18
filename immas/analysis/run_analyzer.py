"""
Analyze a JSONL router run log.

Outputs
-------
- Summary metrics (MAE/RMSE/correlation) for latency + cost + cache ratio
- Cache/KV diagnostics:
  - kvmatch_text proxy vs observed obs_cache_ratio
  - predicted cache ratio vs observed cache ratio
- Top latency outliers (with cache/token context)
- Suspicious cases (high prefix match but low cached tokens, etc.)
- Conversation-ordered CSV (turns_sorted.csv) for debugging KV cache behavior per dialogue.
- Plots (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Mapping, Tuple

from immas.analysis.utils import (
    mean,
    rmse,
    mae,
    pearsonr,
    load_jsonl,
    _f,
    _i,
    _s,
    _pairs,
    _short_id,
    _write_turns_csv,
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

    # Completion-order time series
    ok_by_end = sorted(ok, key=lambda r: _f(r.get("t_end_monotonic")))

    pred_lat, obs_lat = _pairs(ok_by_end, "pred_latency_ms", "obs_latency_ms")
    pred_cost, obs_cost = _pairs(ok_by_end, "pred_cost_tokens", "obs_total_tokens")

    # Cache metrics (router proxy vs backend usage)
    pred_cache, obs_cache = _pairs(ok_by_end, "pred_cache_ratio", "obs_cache_ratio")
    kvmatch, obs_cache2 = _pairs(ok_by_end, "kvmatch_text", "obs_cache_ratio")

    obs_prompt_tokens = [_i(r.get("obs_prompt_tokens")) for r in ok_by_end]
    obs_cached_tokens = [_i(r.get("obs_cached_tokens")) for r in ok_by_end]
    obs_cache_ratio = [_f(r.get("obs_cache_ratio")) for r in ok_by_end]

    correct = [bool(r.get("correct", True)) for r in ok_by_end]
    acc = sum(1 for c in correct if c) / len(correct) if correct else 0.0

    print("\nPrediction accuracy / regression metrics")
    print("---------------------------------------")
    print(f"Accuracy(correct):    {acc:.4f}")
    print(f"Latency MAE (ms):     {mae(pred_lat, obs_lat):.2f}")
    print(f"Latency RMSE (ms):    {rmse(pred_lat, obs_lat):.2f}")
    print(f"Latency corr:         {pearsonr(pred_lat, obs_lat):.3f}")
    print(f"Cost MAE (tok):       {mae(pred_cost, obs_cost):.2f}")
    print(f"Cost RMSE (tok):      {rmse(pred_cost, obs_cost):.2f}")
    print(f"Cost corr:            {pearsonr(pred_cost, obs_cost):.3f}")

    print("\nKV cache / prefix reuse")
    print("-----------------------")
    print(f"Mean obs_prompt_tokens:  {mean([float(x) for x in obs_prompt_tokens]):.1f}")
    print(f"Mean obs_cached_tokens:  {mean([float(x) for x in obs_cached_tokens]):.1f}")
    print(f"Mean obs_cache_ratio:    {mean(obs_cache_ratio):.3f}")
    if pred_cache and obs_cache:
        print(f"CacheRatio MAE:          {mae(pred_cache, obs_cache):.3f}")
        print(f"CacheRatio RMSE:         {rmse(pred_cache, obs_cache):.3f}")
        print(f"CacheRatio corr:         {pearsonr(pred_cache, obs_cache):.3f}")
    if kvmatch and obs_cache2:
        print(f"kvmatch_text corr(obs):  {pearsonr(kvmatch, obs_cache2):.3f}")

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

    # Check turn continuity per dialogue_id
    by_did: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in ok_by_turn:
        by_did[_s(r.get("dialogue_id"))].append(r)

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

    # Latency outliers
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

    # Suspicious KV cases: high text match but low cached ratio (or vice versa)
    suspicious: List[Mapping[str, Any]] = []
    for r in ok_by_end:
        kv = _f(r.get("kvmatch_text"))
        ocr = _f(r.get("obs_cache_ratio"))
        if (kv >= 0.9 and ocr <= 0.1) or (kv <= 0.1 and ocr >= 0.9):
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

    # Plots (optional)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available; skipping plots.")
        return

    xs = list(range(len(ok_by_end)))

    # Time series: latency
    plt.figure(figsize=(12, 5))
    plt.plot(xs, obs_lat, label="observed latency (ms)", linewidth=1.5)
    plt.plot(xs, pred_lat, label="predicted latency (ms)", linewidth=1.0, alpha=0.8)
    plt.title("Latency over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "latency_timeseries.png", dpi=160)
    plt.close()

    # Time series: cache ratio & kvmatch
    plt.figure(figsize=(12, 5))
    plt.plot(xs, obs_cache_ratio, label="observed cache ratio", linewidth=1.5)
    if pred_cache:
        plt.plot(
            xs, pred_cache, label="predicted cache ratio", linewidth=1.0, alpha=0.8
        )
    if kvmatch:
        plt.plot(xs, kvmatch, label="kvmatch_text (proxy)", linewidth=1.0, alpha=0.8)
    plt.title("KV cache reuse over time (completion order)")
    plt.xlabel("request index (by completion time)")
    plt.ylabel("ratio")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cache_ratio_timeseries.png", dpi=160)
    plt.close()

    # Scatter: predicted vs observed latency
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_lat, obs_lat, s=10, alpha=0.6)
    lo = min(min(pred_lat), min(obs_lat)) if pred_lat and obs_lat else 0
    hi = max(max(pred_lat), max(obs_lat)) if pred_lat and obs_lat else 1
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.title("Predicted vs observed latency")
    plt.xlabel("predicted latency (ms)")
    plt.ylabel("observed latency (ms)")
    plt.tight_layout()
    plt.savefig(outdir / "latency_scatter.png", dpi=160)
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
    if obs_cache_ratio and obs_lat:
        plt.figure(figsize=(6, 6))
        plt.scatter(obs_cache_ratio, obs_lat, s=10, alpha=0.6)
        plt.title("Observed cache ratio vs observed latency")
        plt.xlabel("obs_cache_ratio")
        plt.ylabel("obs_latency_ms")
        plt.xlim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(outdir / "obs_cache_ratio_vs_latency.png", dpi=160)
        plt.close()

    print(f"\nWrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
