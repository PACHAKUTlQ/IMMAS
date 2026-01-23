"""
immas.openai.usage

Parse token usage fields across slightly different OpenAI-compatible schemas.

We support:
- Chat Completions style:
  usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
  usage.prompt_tokens_details.cached_tokens
- Responses/Realtimes style:
  usage.input_tokens, usage.output_tokens, usage.total_tokens
  usage.input_token_details.cached_tokens

Notes
-----
Some backends do not report cached-token accounting at all. In that case, we
still return cached_tokens=0 but mark cached_tokens_known=False so callers can
avoid drawing incorrect conclusions (e.g., false "eviction detected").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _get_mapping(x: Any) -> Mapping[str, Any] | None:
    return x if isinstance(x, Mapping) else None


@dataclass(frozen=True, slots=True)
class ParsedUsage:
    """
    Parsed usage fields with best-effort compatibility.

    Attributes
    ----------
    cached_tokens_known
        True if the backend explicitly reported some cached-token accounting field,
        even if the value is zero. False means "unknown/unreported", not "zero".
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    cached_tokens_known: bool

    @property
    def cache_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        r = float(self.cached_tokens) / float(self.prompt_tokens)
        return max(0.0, min(1.0, r))


def parse_usage(resp_json: Any) -> ParsedUsage:
    """
    Best-effort usage parser.

    Missing fields become zero, but `cached_tokens_known` will be False unless the
    backend explicitly reports a cached-token field.
    """

    usage = _get_mapping(
        resp_json.get("usage") if isinstance(resp_json, Mapping) else None
    )
    if usage is None:
        return ParsedUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cached_tokens=0,
            cached_tokens_known=False,
        )

    # Chat Completions names
    prompt_tokens = _int(usage.get("prompt_tokens"))
    completion_tokens = _int(usage.get("completion_tokens"))
    total_tokens = _int(usage.get("total_tokens"))

    # Responses/Realtimes names (fallback)
    if prompt_tokens == 0:
        prompt_tokens = _int(usage.get("input_tokens"))
    if completion_tokens == 0:
        completion_tokens = _int(usage.get("output_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    cached_tokens = 0
    cached_tokens_known = False

    # Chat Completions: prompt_tokens_details.cached_tokens
    ptd = _get_mapping(usage.get("prompt_tokens_details"))
    if ptd is not None and "cached_tokens" in ptd:
        cached_tokens_known = True
        cached_tokens = max(cached_tokens, _int(ptd.get("cached_tokens")))

    # Responses/Realtimes: input_token_details.cached_tokens
    itd = _get_mapping(usage.get("input_token_details"))
    if itd is not None and "cached_tokens" in itd:
        cached_tokens_known = True
        cached_tokens = max(cached_tokens, _int(itd.get("cached_tokens")))

    # Clamp to something sane
    cached_tokens = max(
        0, min(cached_tokens, prompt_tokens if prompt_tokens > 0 else cached_tokens)
    )

    return ParsedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_known=cached_tokens_known,
    )
