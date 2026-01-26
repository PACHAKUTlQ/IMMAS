"""
immas.analysis.analyzer_cost

Observed cost-proxy computation for analysis.

The router trains/logs `pred_cost_tokens` as a scalar cost proxy (arbitrary units)
that may incorporate backend-specific token pricing:

    cost = input_price * uncached_prompt_tokens
         + cached_input_price * cached_prompt_tokens
         + output_price * completion_tokens

The JSONL router log does not currently store `obs_cost_tokens` directly, but it
does store usage fields sufficient to reconstruct it:
- obs_prompt_tokens
- obs_completion_tokens
- obs_cached_tokens

This module:
- loads per-backend prices from the router YAML config (optional),
- computes `obs_cost_tokens` per record,
- annotates records with that derived field so the rest of analysis can compare
  `pred_cost_tokens` against the correct observed target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from immas.router.config import load_router_app_config
from immas.router.pricing import BackendTokenPrices


@dataclass(frozen=True, slots=True)
class CostAnnotationResult:
    """
    Summary of `annotate_records_with_observed_cost()`.

    Attributes
    ----------
    n_annotated
        Number of records where obs_cost_tokens was set.
    n_skipped
        Number of records skipped due to insufficient token fields.
    n_unknown_backend
        Number of annotated records whose backend_id was missing from the provided
        price table (default prices were used).
    """

    n_annotated: int
    n_skipped: int
    n_unknown_backend: int


def load_backend_prices_from_router_config(path: str) -> dict[str, BackendTokenPrices]:
    """
    Load per-backend token prices from a router YAML config file.

    Parameters
    ----------
    path
        Path to the router YAML config.

    Returns
    -------
    dict[str, BackendTokenPrices]
        Mapping backend_id -> prices.
    """

    cfg = load_router_app_config(str(path))
    return {
        b.backend_id: BackendTokenPrices(
            input_token_price=float(b.input_token_price),
            cached_input_token_price=float(b.cached_input_token_price),
            output_token_price=float(b.output_token_price),
        )
        for b in cfg.backends
    }


def compute_observed_cost_tokens_from_counts(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int,
    prices: BackendTokenPrices,
) -> float:
    """
    Compute the observed cost proxy from token counts and per-backend prices.

    This mirrors `immas.router.pricing.compute_observed_cost_tokens()` but operates on
    raw counts to keep analysis independent from ParsedUsage shape.

    Notes
    -----
    - Negative counts are clamped to 0.
    - cached_prompt_tokens is clamped to [0, prompt_tokens].
    """

    p = prices.normalized()

    pt = max(0, int(prompt_tokens))
    ct = max(0, int(completion_tokens))

    cached = max(0, int(cached_prompt_tokens))
    cached = min(pt, cached)

    uncached_prompt = max(0, pt - cached)

    return (
        float(p.input_token_price) * float(uncached_prompt)
        + float(p.cached_input_token_price) * float(cached)
        + float(p.output_token_price) * float(ct)
    )


def _try_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _derive_token_counts_from_record(
    r: Mapping[str, Any],
) -> tuple[int, int, int] | None:
    """
    Extract (prompt_tokens, completion_tokens, cached_prompt_tokens) from one record.

    This is intentionally defensive to support older logs. If required token counts
    cannot be derived, returns None.
    """

    pt = _try_int(r.get("obs_prompt_tokens"))
    ct = _try_int(r.get("obs_completion_tokens"))
    tt = _try_int(r.get("obs_total_tokens"))
    cached = _try_int(r.get("obs_cached_tokens"))

    # Attempt to derive missing fields from total tokens.
    if ct is None and tt is not None and pt is not None:
        ct = tt - pt
    if pt is None and tt is not None and ct is not None:
        pt = tt - ct

    if pt is None or ct is None:
        return None

    cached_i = int(cached) if cached is not None else 0
    return int(pt), int(ct), int(cached_i)


def annotate_records_with_observed_cost(
    *,
    records: Sequence[MutableMapping[str, Any]],
    prices_by_backend_id: Mapping[str, BackendTokenPrices],
    default_prices: BackendTokenPrices | None = None,
    out_key: str = "obs_cost_tokens",
    overwrite: bool = True,
) -> CostAnnotationResult:
    """
    Annotate JSONL records with derived observed cost proxy.

    Parameters
    ----------
    records
        Records to mutate in-place.
    prices_by_backend_id
        Mapping backend_id -> prices.
    default_prices
        Prices used when backend_id is unknown or missing. If None, uses
        BackendTokenPrices() defaults.
    out_key
        Field name to store the derived cost in.
    overwrite
        If False, do not overwrite out_key when already present.

    Returns
    -------
    CostAnnotationResult
        Summary counts for observability in CLI output.
    """

    dp = default_prices or BackendTokenPrices()

    n_annotated = 0
    n_skipped = 0
    n_unknown_backend = 0

    for r in records:
        if (not overwrite) and (out_key in r):
            continue

        counts = _derive_token_counts_from_record(r)
        if counts is None:
            n_skipped += 1
            continue

        prompt_tokens, completion_tokens, cached_prompt_tokens = counts

        backend_id = str(r.get("backend_id") or "").strip()
        prices = prices_by_backend_id.get(backend_id)
        if prices is None:
            prices = dp
            n_unknown_backend += 1

        r[out_key] = compute_observed_cost_tokens_from_counts(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            prices=prices,
        )
        n_annotated += 1

    return CostAnnotationResult(
        n_annotated=n_annotated,
        n_skipped=n_skipped,
        n_unknown_backend=n_unknown_backend,
    )
