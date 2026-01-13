#!/usr/bin/env python3
"""Summarize multiple one_click runs for router comparison.

Reads baseline/logs/<run_id>/summary.json and prints a compact table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "runs",
        nargs="+",
        help="Run dirs or run_ids under baseline/logs/",
    )
    ap.add_argument(
        "--logs-dir",
        default="baseline/logs",
        help="Logs root (default: baseline/logs)",
    )
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir).resolve()

    rows: List[Dict[str, Any]] = []
    for r in args.runs:
        rd = Path(r)
        if not rd.exists():
            rd = logs_dir / r
        rd = rd.resolve()
        s = _load_summary(rd)
        perf = s.get("chosen_task_performance") or {}
        token = s.get("chosen_token_usage") or {}
        welfare = s.get("chosen_social_welfare") or {}
        rows.append(
            {
                "run_id": s.get("run_id", rd.name),
                "router": s.get("router"),
                "requests": s.get("requests"),
                "error_rate": s.get("error_rate"),
                "perf_mean": perf.get("mean"),
                "perf_p50": perf.get("p50"),
                "perf_p95": perf.get("p95"),
                "perf_count": perf.get("count"),
                "tokens_mean": token.get("mean"),
                "tokens_p50": token.get("p50"),
                "tokens_p95": token.get("p95"),
                "welfare_mean": welfare.get("mean"),
                "welfare_p50": welfare.get("p50"),
                "welfare_p95": welfare.get("p95"),
                "e2e_p50_ms": (s.get("latency_ms") or {}).get("e2e_p50"),
                "chosen_llm_p50_ms": (s.get("latency_ms") or {}).get("chosen_llm_p50"),
            }
        )

    # Print a markdown table
    headers = [
        "run_id",
        "router",
        "requests",
        "error_rate",
        "perf_mean",
        "perf_p50",
        "perf_p95",
        "perf_count",
        "tokens_mean",
        "tokens_p50",
        "tokens_p95",
        "welfare_mean",
        "welfare_p50",
        "welfare_p95",
        "e2e_p50_ms",
        "chosen_llm_p50_ms",
    ]

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")


if __name__ == "__main__":
    main()
