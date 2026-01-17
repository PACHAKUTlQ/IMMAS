"""
immas.openai.chat

Chat message helpers:
- deterministic serialization of messages (for router-side prefix matching)
- assistant message extraction from Chat Completions responses
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _content_to_text(content: Any) -> str:
    """
    Convert OpenAI 'content' into a stable string.

    We primarily expect `str` in this project. If content is a structured list
    (multimodal), we JSON-encode it deterministically.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return str(content)


def serialize_chat_messages(messages: Any) -> str:
    """
    Deterministically serialize chat messages into an append-only text form.

    Important property:
    - if `messages_t` is a prefix (by elements) of `messages_{t+1}`,
      then `serialize(messages_t)` is a byte-for-byte prefix of
      `serialize(messages_{t+1})`.

    This makes it suitable for router-side LCP (longest common prefix) matching.
    """
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for m in messages:
        if not isinstance(m, Mapping):
            continue
        role = str(m.get("role") or "")
        name = m.get("name")
        content = _content_to_text(m.get("content"))

        # Include role (and optional name) with hard separators.
        if name is None:
            parts.append(f"<<role:{role}>>\n")
        else:
            parts.append(f"<<role:{role};name:{name}>>\n")
        parts.append(content)
        parts.append("\n<<end>>\n")

    return "".join(parts)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Assistant message extracted from a Chat Completions response."""

    role: str
    content: str

    def to_openai_message(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


def extract_first_assistant_message(resp_json: Any) -> Optional[AssistantMessage]:
    """
    Extract choices[0].message.{role,content} from a Chat Completions response.

    Returns None if not present (tool calls, errors, non-standard responses).
    """
    if not isinstance(resp_json, Mapping):
        return None
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if not isinstance(c0, Mapping):
        return None

    msg = c0.get("message")
    if not isinstance(msg, Mapping):
        return None

    role = str(msg.get("role") or "assistant")
    content = msg.get("content")
    if content is None:
        # e.g. tool call response
        return AssistantMessage(role=role, content="")
    return AssistantMessage(role=role, content=_content_to_text(content))
