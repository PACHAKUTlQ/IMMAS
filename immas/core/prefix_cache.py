"""
A tiny text-based prefix cache used only for feature construction (kvmatch).

We intentionally do *character* prefix matching:
- No tokenization complexity
- Works across models with different tokenizers (as requested)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


CacheKey = Tuple[str, str]  # (model_name, dialogue_id)


def common_prefix_length(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    n = min(len(a), len(b))
    i = 0
    # Tight loop; good enough for basic experiments.
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TextPrefixCache:
    """
        Cache the last "known prefix text" per (model, dialogue_id).

        In this basic version:
    - We only track per-dialogue prefix, because cross-dialogue prefix reuse is rare.
    - We do not implement eviction; later can add LRU/size constraints.
    """

    def __init__(self) -> None:
        self._cache: Dict[CacheKey, str] = {}

    def get(self, *, model: str, dialogue_id: str) -> Optional[str]:
        """Return cached prefix text or None if absent."""
        return self._cache.get((model, dialogue_id))

    def update(self, *, model: str, dialogue_id: str, cached_text: str) -> None:
        """Set cached prefix text."""
        self._cache[(model, dialogue_id)] = cached_text

    def match_ratio(self, *, model: str, dialogue_id: str, prompt_text: str) -> float:
        """
        Compute prefix match ratio: LCP(cached_text, prompt_text) / len(prompt_text).

        Returns 0.0 if no cached entry exists.
        """
        if not prompt_text:
            return 0.0

        cached = self.get(model=model, dialogue_id=dialogue_id)
        if not cached:
            return 0.0

        lcp = common_prefix_length(cached, prompt_text)
        return float(lcp) / float(len(prompt_text))
