"""
immas.analysis.analyzer_welfare

Welfare annotation and auction-parameter loading for analysis.

This module reconstructs welfare-related quantities for router JSONL records:

- Predicted welfare components (decision-time):
    pred_client_valuation
    pred_base_cost
    pred_welfare

- Observed welfare components (post-hoc, analysis-time):
    obs_client_valuation
    obs_base_cost
    obs_welfare

- Best-available predicted welfare (from per-request backend_scores, if present):
    best_pred_welfare
    pred_welfare_regret := best_pred_welfare - pred_welfare

Notes
-----
- Predicted values are taken from the log fields `chosen_*` when present
  (auction routing), otherwise reconstructed using router auction parameters.
- Observed welfare uses:
    P := correct (0/1)
    L := obs_latency_ms
    C := obs_cost_tokens (derived cost proxy; see analyzer_cost.py)
  and applies the same auction scales/δ as the router configuration.

- This module is analysis-only; it mutates records in-place for downstream
  reporting/plotting.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from immas.analysis.utils import _try_bool, _try_float, _finite_or_none
from immas.router.auction.mechanism import (
    AuctionParams,
    compute_client_valuation,
    compute_scaled_base_cost,
    compute_welfare,
)
from immas.router.config import load_router_app_config


@dataclass(frozen=True, slots=True)
class WelfareAnnotationResult:
    """
    Summary of `annotate_records_with_welfare()`.

    Attributes
    ----------
    n_pred_annotated
        Number of records where pred_* welfare fields were set.
    n_obs_annotated
        Number of records where obs_* welfare fields were set.
    n_regret_annotated
        Number of records where best_pred_welfare and regret were set.
    n_skipped
        Number of records skipped due to missing required fields.
    """

    n_pred_annotated: int
    n_obs_annotated: int
    n_regret_annotated: int
    n_skipped: int


def load_auction_params_from_router_config(path: str) -> AuctionParams:
    """
    Load auction welfare parameters from a router YAML config file.

    Parameters
    ----------
    path
        Path to the router YAML config.

    Returns
    -------
    AuctionParams
        Parameters matching router runtime welfare computation.
    """

    cfg = load_router_app_config(str(path))
    a = cfg.router.auction
    return AuctionParams(
        quality_scale=float(a.quality_scale),
        latency_scale=float(a.latency_scale),
        cost_scale=float(a.cost_scale),
        delta_default=float(a.delta_default),
        min_welfare_edge=float(a.min_welfare_edge),
        mcmf_scale=int(a.mcmf_scale),
    )


def _best_pred_welfare_from_backend_scores(r: Mapping[str, Any]) -> float | None:
    """
    Extract max predicted welfare from record.backend_scores if available.

    The JSONL log stores backend_scores as a list of dicts (from dataclasses.asdict).
    """

    bs = r.get("backend_scores")
    if not isinstance(bs, list) or not bs:
        return None

    best: float | None = None
    for item in bs:
        if not isinstance(item, Mapping):
            continue
        w = _finite_or_none(_try_float(item.get("welfare")))
        if w is None:
            continue
        if best is None or w > best:
            best = float(w)

    return best


def annotate_records_with_welfare(
    *,
    records: Sequence[MutableMapping[str, Any]],
    params: AuctionParams,
    out_pred_client_val_key: str = "pred_client_valuation",
    out_pred_base_cost_key: str = "pred_base_cost",
    out_pred_welfare_key: str = "pred_welfare",
    out_obs_client_val_key: str = "obs_client_valuation",
    out_obs_base_cost_key: str = "obs_base_cost",
    out_obs_welfare_key: str = "obs_welfare",
    out_best_pred_welfare_key: str = "best_pred_welfare",
    out_regret_key: str = "pred_welfare_regret",
    overwrite: bool = True,
) -> WelfareAnnotationResult:
    """
    Annotate JSONL records with welfare-related derived fields.

    Parameters
    ----------
    records
        Records to mutate in-place.
    params
        Auction parameters (scales, δ, etc).
    out_*_key
        Output field names for derived values.
    overwrite
        If False, do not overwrite existing keys.

    Returns
    -------
    WelfareAnnotationResult
        Summary counts.
    """

    n_pred_annotated = 0
    n_obs_annotated = 0
    n_regret_annotated = 0
    n_skipped = 0

    delta = float(params.delta_default)

    for r in records:
        if not isinstance(r, MutableMapping):
            n_skipped += 1
            continue

        # Predicted welfare
        can_write_pred = overwrite or (
            out_pred_client_val_key not in r
            and out_pred_base_cost_key not in r
            and out_pred_welfare_key not in r
        )

        if can_write_pred:
            # Prefer log-native auction fields if present (authoritative).
            chosen_val = _finite_or_none(_try_float(r.get("chosen_client_valuation")))
            chosen_base = _finite_or_none(_try_float(r.get("chosen_base_cost")))
            chosen_w = _finite_or_none(_try_float(r.get("chosen_welfare")))

            if (
                chosen_val is not None
                and chosen_base is not None
                and chosen_w is not None
            ):
                r[out_pred_client_val_key] = float(chosen_val)
                r[out_pred_base_cost_key] = float(chosen_base)
                r[out_pred_welfare_key] = float(chosen_w)
                n_pred_annotated += 1
            else:
                pl = _finite_or_none(_try_float(r.get("pred_latency_ms")))
                pc = _finite_or_none(_try_float(r.get("pred_cost_tokens")))
                pp = _finite_or_none(_try_float(r.get("pred_perf_prob")))

                if pl is not None and pc is not None and pp is not None:
                    w, val, base = compute_welfare(
                        delta=float(delta),
                        pred_latency_ms=float(pl),
                        pred_cost_tokens=float(pc),
                        pred_perf_prob=float(pp),
                        params=params,
                    )
                    r[out_pred_client_val_key] = float(val)
                    r[out_pred_base_cost_key] = float(base)
                    r[out_pred_welfare_key] = float(w)
                    n_pred_annotated += 1

        # Observed welfare (post-hoc)
        can_write_obs = overwrite or (
            out_obs_client_val_key not in r
            and out_obs_base_cost_key not in r
            and out_obs_welfare_key not in r
        )

        if can_write_obs:
            ol = _finite_or_none(_try_float(r.get("obs_latency_ms")))
            oc = _finite_or_none(_try_float(r.get("obs_cost_tokens")))
            corr = _try_bool(r.get("correct", True), default=True)

            if ol is not None and oc is not None:
                # Observed performance proxy: P := correct (0/1).
                val_obs = compute_client_valuation(
                    delta=float(delta),
                    pred_latency_ms=float(ol),
                    pred_perf_prob=1.0 if corr else 0.0,
                    params=params,
                )
                base_obs = compute_scaled_base_cost(
                    pred_cost_tokens=float(oc),
                    params=params,
                )
                r[out_obs_client_val_key] = float(val_obs)
                r[out_obs_base_cost_key] = float(base_obs)
                r[out_obs_welfare_key] = float(val_obs - base_obs)
                n_obs_annotated += 1

        # Regret relative to best predicted welfare (per-request)
        can_write_regret = overwrite or (
            out_best_pred_welfare_key not in r and out_regret_key not in r
        )
        if can_write_regret:
            best = _best_pred_welfare_from_backend_scores(r)
            pred_w = _finite_or_none(_try_float(r.get(out_pred_welfare_key)))
            if best is not None and pred_w is not None:
                r[out_best_pred_welfare_key] = float(best)
                r[out_regret_key] = float(best - pred_w)
                n_regret_annotated += 1

    return WelfareAnnotationResult(
        n_pred_annotated=int(n_pred_annotated),
        n_obs_annotated=int(n_obs_annotated),
        n_regret_annotated=int(n_regret_annotated),
        n_skipped=int(n_skipped),
    )
