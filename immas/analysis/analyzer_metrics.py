"""
immas.analysis.analyzer_metrics

Reusable metric helpers for analyzer reporting.

Currently includes:
- binary probabilistic prediction metrics (for pred_perf_prob vs correct)
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Iterable, Sequence

from immas.analysis.utils import _clamp01, _is_finite


@dataclass(frozen=True, slots=True)
class BinaryProbMetrics:
    """
    Summary metrics for probabilistic binary predictions.

    Attributes
    ----------
    n
        Number of usable pairs.
    mean_pred
        Mean predicted probability.
    mean_obs
        Mean observed label (i.e. empirical positive rate).
    accuracy_at_0_5
        Accuracy when predicting True iff p >= 0.5.
    brier
        Brier score: mean (p - y)^2.
    log_loss
        Log loss (cross entropy), with numerical clipping.
    """

    n: int
    mean_pred: float
    mean_obs: float
    accuracy_at_0_5: float
    brier: float
    log_loss: float


def compute_binary_prob_metrics(
    pred_probs: Sequence[float],
    obs_labels: Sequence[bool],
    *,
    eps: float = 1e-12,
) -> BinaryProbMetrics:
    """
    Compute metrics for probabilistic predictions.

    Notes
    -----
    - Inputs are aligned by zip; caller should already filter missing values.
    - Probabilities are clamped to [0, 1].
    - Log loss uses clipping to [eps, 1-eps].
    """

    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")

    n = min(len(pred_probs), len(obs_labels))
    if n <= 0:
        return BinaryProbMetrics(
            n=0,
            mean_pred=0.0,
            mean_obs=0.0,
            accuracy_at_0_5=0.0,
            brier=0.0,
            log_loss=0.0,
        )

    ps: list[float] = []
    ys: list[float] = []
    for p, yb in zip(pred_probs[:n], obs_labels[:n]):
        pf = float(p)
        if not _is_finite(pf):
            continue
        ps.append(_clamp01(pf))
        ys.append(1.0 if bool(yb) else 0.0)

    if not ps:
        return BinaryProbMetrics(
            n=0,
            mean_pred=0.0,
            mean_obs=0.0,
            accuracy_at_0_5=0.0,
            brier=0.0,
            log_loss=0.0,
        )

    mean_pred = float(sum(ps) / len(ps))
    mean_obs = float(sum(ys) / len(ys))

    correct = 0
    brier_sum = 0.0
    ll_sum = 0.0

    for p, y in zip(ps, ys):
        yb = bool(y >= 0.5)
        pred = bool(p >= 0.5)
        correct += 1 if pred == yb else 0

        diff = p - y
        brier_sum += diff * diff

        p_clip = min(1.0 - eps, max(eps, p))
        ll_sum += -(y * math.log(p_clip) + (1.0 - y) * math.log(1.0 - p_clip))

    n2 = len(ps)
    return BinaryProbMetrics(
        n=n2,
        mean_pred=mean_pred,
        mean_obs=mean_obs,
        accuracy_at_0_5=float(correct) / float(n2),
        brier=brier_sum / float(n2),
        log_loss=ll_sum / float(n2),
    )


def extract_perf_pairs_from_records(
    records: Iterable[dict],
    *,
    pred_key: str = "pred_perf_prob",
    obs_key: str = "correct",
) -> tuple[list[float], list[bool]]:
    """
    Extract aligned (pred_prob, correct_bool) pairs from JSONL records.

    Missing/unparseable pred_prob values are skipped.
    Missing obs labels default to True (to match router logging semantics).
    """

    ps: list[float] = []
    ys: list[bool] = []

    for r in records:
        if not isinstance(r, dict):
            continue
        pv = r.get(pred_key)
        if pv is None:
            continue
        try:
            p = float(pv)
        except Exception:
            continue
        if not _is_finite(p):
            continue

        y = bool(r.get(obs_key, True))
        ps.append(_clamp01(p))
        ys.append(y)

    return ps, ys
