"""
immas.router.prefix_cache

A router-side text prefix cache for computing KV-cache proxy features.

We maintain the last seen serialized prompt text per (backend, model, dialogue_id).
Given a new prompt, we compute the longest common prefix (LCP) against the cached
text and expose:
- lcp_chars: number of matching prefix characters
- ratio: lcp_chars / len(prompt_text)

This ratio is a proxy for prompt prefix reuse, which should correlate with
backend-reported cached prompt tokens.

Eviction
--------
Backends like vLLM may evict prompt-cache entries independently of the router.
The router can optionally evict its own record when it detects a likely backend
cache miss (e.g. near-perfect text prefix match but reported cached_tokens ~ 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


CacheKey = Tuple[str, str, str]  # (backend_id, model, dialogue_id)


def common_prefix_length(a: str, b: str) -> int:
    """Return the length (in characters) of the longest common prefix of a and b."""

    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """Detailed prefix match statistics for one prompt."""

    ratio: float
    lcp_chars: int
    prompt_chars: int
    cached_chars: int


class TextPrefixCache:
    """Stores last known prefix text per (backend, model, dialogue_id)."""

    def __init__(self) -> None:
        self._cache: Dict[CacheKey, str] = {}

    def get(self, *, backend_id: str, model: str, dialogue_id: str) -> Optional[str]:
        """Fetch cached text for this key, if present."""

        return self._cache.get((backend_id, model, dialogue_id))

    def update(
        self, *, backend_id: str, model: str, dialogue_id: str, cached_text: str
    ) -> None:
        """Insert/overwrite cached text for this key."""

        self._cache[(backend_id, model, dialogue_id)] = cached_text

    def evict(self, *, backend_id: str, model: str, dialogue_id: str) -> bool:
        """
        Evict cached text for this key.

        Returns
        -------
        bool
            True if an entry existed and was removed; False otherwise.
        """

        key: CacheKey = (backend_id, model, dialogue_id)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def match(
        self, *, backend_id: str, model: str, dialogue_id: str, prompt_text: str
    ) -> PrefixMatch:
        """
        Compute prefix-match stats against the cached text for this key.

        Notes
        -----
        - ratio is defined as lcp_chars / prompt_chars (0 if prompt_text is empty).
        - cached text may be shorter than prompt (common in multi-turn chat),
          in which case the best-case lcp equals cached_chars.
        """

        prompt_chars = len(prompt_text)
        cached = self.get(backend_id=backend_id, model=model, dialogue_id=dialogue_id)
        cached_chars = len(cached) if cached is not None else 0

        if prompt_chars <= 0 or not cached:
            return PrefixMatch(
                ratio=0.0,
                lcp_chars=0,
                prompt_chars=prompt_chars,
                cached_chars=cached_chars,
            )

        lcp = common_prefix_length(cached, prompt_text)
        ratio = float(lcp) / float(prompt_chars)
        ratio = max(0.0, min(1.0, ratio))

        return PrefixMatch(
            ratio=ratio,
            lcp_chars=int(lcp),
            prompt_chars=int(prompt_chars),
            cached_chars=int(cached_chars),
        )

    def match_ratio(
        self, *, backend_id: str, model: str, dialogue_id: str, prompt_text: str
    ) -> float:
        """Backward-compatible convenience wrapper returning only the ratio."""

        return self.match(
            backend_id=backend_id,
            model=model,
            dialogue_id=dialogue_id,
            prompt_text=prompt_text,
        ).ratio
