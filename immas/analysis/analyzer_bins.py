"""
immas.analysis.analyzer_bins

Binning helpers for calibration curves and cache-benefit visualizations.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from immas.analysis.utils import _is_finite


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
