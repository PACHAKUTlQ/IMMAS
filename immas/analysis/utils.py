"""
immas.analysis.utils

Utility functions for analysis scripts.
"""

import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def rmse(pred: Sequence[float], obs: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return math.sqrt(mean([(p - o) ** 2 for p, o in zip(pred, obs)]))


def mae(pred: Sequence[float], obs: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return mean([abs(p - o) for p, o in zip(pred, obs)])


def pearsonr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return float(num / (denx * deny))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(dict(json.loads(line)))
    return out


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _s(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def _pairs(
    records: Iterable[Mapping[str, Any]], pred_key: str, obs_key: str
) -> Tuple[List[float], List[float]]:
    pred: List[float] = []
    obs: List[float] = []
    for r in records:
        if pred_key not in r or obs_key not in r:
            continue
        pred.append(_f(r.get(pred_key)))
        obs.append(_f(r.get(obs_key)))
    return pred, obs


def _short_id(dialogue_id: str, n: int = 12) -> str:
    return dialogue_id if len(dialogue_id) <= n else dialogue_id[:n]


def _csv_fmt(val: Any) -> str:
    """Format floats to 3 decimals, otherwise stringify."""

    if isinstance(val, float):
        return f"{val:.3f}"

    return str(val) if val is not None else ""


def _write_turns_csv(*, out_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write a conversation-ordered CSV (sorted by dialogue_id, then turn_number, then t_start_monotonic).

    This is the main debug artifact for checking:
    - prompt_tokens progression
    - cached_tokens and cache ratio
    - whether cache reuse aligns with kvmatch_text
    """

    cols = [
        "run_id",
        "backend_id",
        "model",
        "source",
        "dialogue_id",
        "turn_number",
        "t_start_monotonic",
        "t_end_monotonic",
        "prompt_chars",
        "cached_prompt_chars",
        "kvmatch_lcp_chars",
        "kvmatch_text",
        "pred_cache_ratio",
        "obs_prompt_tokens",
        "obs_cached_tokens",
        "obs_cache_ratio",
        "pred_latency_ms",
        "obs_latency_ms",
        "pred_cost_tokens",
        "obs_total_tokens",
        "router_inflight",
        "router_rps_1s",
        "error",
        "completion_id",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {c: _csv_fmt(r.get(c)) for c in cols}
            w.writerow(row)
