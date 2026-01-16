"""
immas.common.text

Text helpers shared by router and client.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


def inline_for_cache(text: str) -> str:
    """Normalize answer into a single line to match the prompt formatter behavior."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def extract_last_user_text(messages: Any) -> str:
    """
    Extract the last user message content from an OpenAI chat request.

    Returns empty string on schema mismatch.
    """
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, Mapping) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""
