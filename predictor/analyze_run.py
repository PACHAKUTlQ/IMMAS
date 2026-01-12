"""
Analyze a JSONL run log produced by client_coqa_demo.py.

Outputs:
- Summary metrics (MAE/RMSE/correlation) for latency + cost
- Top latency outliers
- Plots (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rmse(pred: List[float], obs: List[float]) -> float:
    if not pred:
        return 0.0
    return math.sqrt(mean([(p - o) ** 2 for p, o in zip(pred, obs)]))


def mae(pred: List[float], obs: List[float]) -> float:
    if not pred:
        return 0.0
    return mean([abs(p - o) for p, o in zip(pred, obs)])


def pearsonr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get("RUN_LOG_PATH", "coqa_run.jsonl"))
    ap.add_argument("--outdir", default="run_analysis")
    ap.add_argument(
        "--run-id", default="", help="If provided, filter to a specific run_id."
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    records = load_jsonl(log_path)

    if args.run_id:
        records = [r for r in records if str(r.get("run_id", "")) == args.run_id]

    ok = [r for r in records if not r.get("error")]
    err = [r for r in records if r.get("error")]

    print(f"Loaded: {len(records)} records   ok={len(ok)}   errors={len(err)}")

    if not ok:
        print("No successful records to analyze.")
        return

    # Sort by completion time for time-series plots
    ok.sort(key=lambda r: float(r.get("t_end_monotonic", 0.0)))

    pred_lat = [float(r["pred_latency_ms"]) for r in ok]
    obs_lat = [float(r["obs_latency_ms"]) for r in ok]

    pred_cost = [float(r["pred_cost_tokens"]) for r in ok]
    obs_cost = [float(r["obs_total_tokens"]) for r in ok]

    correct = [bool(r["correct"]) for r in ok]
    acc = sum(1 for c in correct if c) / len(correct)

    print("\nPrediction accuracy")
    print("-------------------")
    print(f"Accuracy(correct): {acc:.4f}")
    print(f"Latency MAE (ms):  {mae(pred_lat, obs_lat):.2f}")
    print(f"Latency RMSE (ms): {rmse(pred_lat, obs_lat):.2f}")
    print(f"Latency corr:      {pearsonr(pred_lat, obs_lat):.3f}")
    print(f"Cost MAE (tok):    {mae(pred_cost, obs_cost):.2f}")
    print(f"Cost RMSE (tok):   {rmse(pred_cost, obs_cost):.2f}")
    print(f"Cost corr:         {pearsonr(pred_cost, obs_cost):.3f}")

    # Outliers
    topk = sorted(ok, key=lambda r: float(r["obs_latency_ms"]), reverse=True)[:10]
    print("\nTop latency outliers")
    print("--------------------")
    for r in topk:
        did = str(r.get("dialogue_id", ""))[:12]
        turn = int(r.get("turn_number", -1))
        obs = float(r.get("obs_latency_ms", 0.0))
        pred = float(r.get("pred_latency_ms", 0.0))
        u = r.get("srv_utilization", None)
        stall_s = r.get("srv_sim_stall_s", None)
        print(
            f"did={did} turn={turn} obs_ms={obs:.1f} pred_ms={pred:.1f} "
            f"srv_u={u if u is not None else 'NA'} stall_s={
                stall_s if stall_s is not None else 'NA'
            }"
        )

    # Plots (optional)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("\nmatplotlib not available; skipping plots.")
        return

    xs = list(range(len(ok)))

    # Time series: predicted vs observed latency
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

    # Scatter: predicted vs observed latency
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_lat, obs_lat, s=10, alpha=0.6)
    lo = min(min(pred_lat), min(obs_lat))
    hi = max(max(pred_lat), max(obs_lat))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.title("Predicted vs observed latency")
    plt.xlabel("predicted latency (ms)")
    plt.ylabel("observed latency (ms)")
    plt.tight_layout()
    plt.savefig(outdir / "latency_scatter.png", dpi=160)
    plt.close()

    # Utilization vs observed latency (if server debug present)
    u = [r.get("srv_utilization", None) for r in ok]
    if any(v is not None for v in u):
        uu = [float(v) if v is not None else float("nan") for v in u]
        stall = [float(r.get("srv_sim_stall_s") or 0.0) for r in ok]

        plt.figure(figsize=(7, 5))
        # Color by stall presence
        colors = ["red" if s > 0 else "blue" for s in stall]
        plt.scatter(uu, obs_lat, s=10, alpha=0.6, c=colors)
        plt.title("Server utilization vs observed latency (red=stall)")
        plt.xlabel("server utilization")
        plt.ylabel("observed latency (ms)")
        plt.tight_layout()
        plt.savefig(outdir / "util_vs_latency.png", dpi=160)
        plt.close()

    print(f"\nWrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
