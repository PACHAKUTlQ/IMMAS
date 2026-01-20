"""
immas.analysis.analyzer_reports

CSV report writers for analyzer outputs.
"""

from __future__ import annotations

import csv

from pathlib import Path
from typing import Any, Dict, Sequence

from immas.analysis.analyzer_series import _is_finite, _pearsonr_finite
from immas.analysis.analyzer_types import DialogueSeries
from immas.analysis.utils import mean, quantile


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
