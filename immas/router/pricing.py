"""
immas.router.pricing

Backend token pricing and cost computation.

This module provides a small, explicit abstraction for per-backend token pricing
and for converting backend-reported usage into a single scalar "cost proxy".

Important
---------
The router intentionally treats prices as arbitrary per-token units; only relative
values matter for routing decisions (since router.auction.cost_scale can further
rescale cost inside welfare).

Given:
- prompt tokens (input),
- cached prompt tokens (cache hit part of input),
- completion tokens (output),

we compute:

    cost = input_price * (prompt_tokens - cached_prompt_tokens)
         + cached_input_price * cached_prompt_tokens
         + output_price * completion_tokens

If cached token accounting is unavailable, cached_prompt_tokens is treated as 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from immas.openai.usage import ParsedUsage


@dataclass(frozen=True, slots=True)
class BackendTokenPrices:
    """
    Per-backend token pricing in arbitrary per-token units.

    Defaults preserve old behavior: cost is roughly proportional to total tokens
    and cache hits do not change cost.
    """

    input_token_price: float = 1.0
    cached_input_token_price: float = 1.0
    output_token_price: float = 1.0

    def normalized(self) -> "BackendTokenPrices":
        """
        Return a non-negative normalized copy (defensive).

        This is intentionally forgiving: negative values are clamped to 0.
        """
        return BackendTokenPrices(
            input_token_price=max(0.0, float(self.input_token_price)),
            cached_input_token_price=max(0.0, float(self.cached_input_token_price)),
            output_token_price=max(0.0, float(self.output_token_price)),
        )


def compute_observed_cost_tokens(
    *, usage: ParsedUsage, prices: BackendTokenPrices
) -> float:
    """
    Compute the router's observed cost proxy from backend-reported usage.

    Parameters
    ----------
    usage
        Parsed usage record including prompt/completion tokens and cached token info.
    prices
        Per-backend token prices.

    Returns
    -------
    float
        Price-weighted token cost proxy (arbitrary units).
    """

    p = prices.normalized()

    prompt_tokens = max(0, int(getattr(usage, "prompt_tokens", 0) or 0))
    completion_tokens = max(0, int(getattr(usage, "completion_tokens", 0) or 0))

    cached_known = bool(getattr(usage, "cached_tokens_known", False))
    cached_tokens_raw = (
        int(getattr(usage, "cached_tokens", 0) or 0) if cached_known else 0
    )
    cached_tokens = max(0, min(prompt_tokens, cached_tokens_raw))

    uncached_prompt = max(0, prompt_tokens - cached_tokens)

    return (
        float(p.input_token_price) * float(uncached_prompt)
        + float(p.cached_input_token_price) * float(cached_tokens)
        + float(p.output_token_price) * float(completion_tokens)
    )
