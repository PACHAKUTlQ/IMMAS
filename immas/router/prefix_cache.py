"""
immas.router.prefix_cache

A router-side text prefix cache for computing kvmatch proxy features.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


CacheKey = Tuple[str, str, str]  # (backend_id, model, dialogue_id)


def common_prefix_length(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TextPrefixCache:
    """Stores last known prefix text per (backend, model, dialogue_id)."""

    def __init__(self) -> None:
        self._cache: Dict[CacheKey, str] = {}

    def get(self, *, backend_id: str, model: str, dialogue_id: str) -> Optional[str]:
        return self._cache.get((backend_id, model, dialogue_id))

    def update(
        self, *, backend_id: str, model: str, dialogue_id: str, cached_text: str
    ) -> None:
        self._cache[(backend_id, model, dialogue_id)] = cached_text

    def match_ratio(
        self, *, backend_id: str, model: str, dialogue_id: str, prompt_text: str
    ) -> float:
        if not prompt_text:
            return 0.0
        cached = self.get(backend_id=backend_id, model=model, dialogue_id=dialogue_id)
        if not cached:
            return 0.0
        lcp = common_prefix_length(cached, prompt_text)
        return float(lcp) / float(len(prompt_text))
