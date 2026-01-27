"""
immas.analysis.utils

Utility functions for analysis scripts.

Notes
-----
- Metrics are computed on the aligned prefix of sequences when lengths differ:
  for sequences xs and ys, we use n = min(len(xs), len(ys)).
"""

from __future__ import annotations

import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _try_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _try_bool(x: Any, *, default: bool = False) -> bool:
    try:
        return bool(x)
    except Exception:
        return bool(default)


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def quantile(xs: Sequence[float], q: float) -> float:
    """
    Compute an interpolated quantile.

    Parameters
    ----------
    q
        In [0, 1]. Values outside are clamped.

    Returns 0.0 on empty input.
    """

    if not xs:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return float(ys[0])

    pos = q * float(len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    frac = pos - float(lo)
    return float((1.0 - frac) * ys[lo] + frac * ys[hi])


def rmse(pred: Sequence[float], obs: Sequence[float]) -> float:
    n = min(len(pred), len(obs))
    if n <= 0:
        return 0.0
    return math.sqrt(
        mean([(float(p) - float(o)) ** 2 for p, o in zip(pred[:n], obs[:n])])
    )


def mae(pred: Sequence[float], obs: Sequence[float]) -> float:
    n = min(len(pred), len(obs))
    if n <= 0:
        return 0.0
    return mean([abs(float(p) - float(o)) for p, o in zip(pred[:n], obs[:n])])


def pearsonr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xsn = [float(x) for x in xs[:n]]
    ysn = [float(y) for y in ys[:n]]

    mx = mean(xsn)
    my = mean(ysn)
    num = sum((x - mx) * (y - my) for x, y in zip(xsn, ysn))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xsn))
    deny = math.sqrt(sum((y - my) ** 2 for y in ysn))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return float(num / (denx * deny))


def r2_score(pred: Sequence[float], obs: Sequence[float]) -> float:
    """
    Compute R^2 on aligned pairs.

    Returns 0.0 if undefined (n < 2 or zero variance in obs).
    """

    n = min(len(pred), len(obs))
    if n < 2:
        return 0.0

    o = [float(x) for x in obs[:n]]
    p = [float(x) for x in pred[:n]]
    mo = mean(o)
    ss_tot = sum((x - mo) ** 2 for x in o)
    if ss_tot <= 0.0:
        return 0.0
    ss_res = sum((x - y) ** 2 for x, y in zip(o, p))
    return float(1.0 - (ss_res / ss_tot))


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
    """
    Extract aligned (pred, obs) float pairs from records.

    This function skips records where either value is missing (None) or not
    convertible to float, to avoid silently injecting zeros.
    """

    pred: List[float] = []
    obs: List[float] = []
    for r in records:
        if pred_key not in r or obs_key not in r:
            continue

        pv = r.get(pred_key)
        ov = r.get(obs_key)
        if pv is None or ov is None:
            continue

        try:
            p = float(pv)
            o = float(ov)
        except Exception:
            continue

        if math.isnan(p) or math.isnan(o) or math.isinf(p) or math.isinf(o):
            continue

        pred.append(p)
        obs.append(o)

    return pred, obs


def _short_id(dialogue_id: str, n: int = 12) -> str:
    return dialogue_id if len(dialogue_id) <= n else dialogue_id[:n]


def _csv_fmt(val: Any) -> str:
    """Format floats to 3 decimals, otherwise stringify (blank for NaN/Inf)."""

    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return ""
        return f"{val:.3f}"
    return str(val) if val is not None else ""


def _write_turns_csv(*, out_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write a conversation-ordered CSV (sorted by dialogue_id, then turn_number, then t_start_monotonic).

    This is the main debug artifact for checking:
    - prompt_tokens progression
    - cached_tokens and cache ratio
    - whether cache reuse aligns with kvmatch_text
    - cost proxy behavior (pred_cost_tokens vs obs_cost_tokens)
    - welfare/payment behavior (pred/obs welfare; VCG fees/payments)
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
        "batch_id",
        "batch_size",
        "prompt_chars",
        "cached_prompt_chars",
        "kvmatch_lcp_chars",
        "kvmatch_text",
        "pred_cache_ratio",
        "obs_prompt_tokens",
        "obs_completion_tokens",
        "obs_cached_tokens",
        "obs_cache_ratio",
        "pred_latency_ms",
        "obs_latency_ms",
        "pred_cost_tokens",
        "obs_cost_tokens",
        "obs_total_tokens",
        "pred_perf_prob",
        "correct",
        "pred_client_valuation",
        "pred_base_cost",
        "pred_welfare",
        "obs_client_valuation",
        "obs_base_cost",
        "obs_welfare",
        "best_pred_welfare",
        "pred_welfare_regret",
        "routing_policy",
        "auction_matched",
        "auction_total_welfare",
        "vcg_fee",
        "vcg_total_payment",
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
