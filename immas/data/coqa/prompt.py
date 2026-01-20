"""
immas.data.coqa.prompt

Prompt formatting/parsing shared by server and client.

Key goals:
- Deterministic prompt format so the fake API can identify (dialogue_id, turn).
- Multi-turn: full story + full history for every request.
- Ends with 'A{turn}:' (no answer included) so that after we append the answer,
  the next prompt is a strict prefix extension -> high text prefix match.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import List, Sequence, Tuple


HistoryTurn = Tuple[str, str]


@dataclass(frozen=True, slots=True)
class ParsedCoqaPrompt:
    """Minimal parsed data needed by the fake API server."""

    dialogue_id: str
    turn_number: int  # 1-based


class CoqaPromptFormatter:
    """
    Formatter for CoQA multi-turn QA prompts.

    The format is intentionally easy to parse:

    ### COQA_DIALOGUE_ID: <id>
    ### COQA_SOURCE: <source>
    ### STORY
    <story>
    ### CONVERSATION
    Q1: ...
    A1: ...
    ...
    Qk: ...
    Ak:
    """

    DIALOGUE_ID_LINE_PREFIX = "### COQA_DIALOGUE_ID: "
    SOURCE_LINE_PREFIX = "### COQA_SOURCE: "
    STORY_MARKER = "### STORY"
    CONVERSATION_MARKER = "### CONVERSATION"

    @classmethod
    def format_turn(
        cls,
        *,
        dialogue_id: str,
        source: str,
        story: str,
        history: Sequence[HistoryTurn],
        question: str,
        turn_number: int,
    ) -> str:
        """
        Format a single turn request.

        Parameters
        ----------
        history
            Sequence of (question, answer) for turns 1..turn_number-1.
        question
            Current turn question.
        turn_number
            1-based.
        """
        if turn_number < 1:
            raise ValueError(f"turn_number must be >= 1, got {turn_number}")

        # Keep Q/A lines single-line to avoid confusing the parser.
        # The story may be multi-paragraph; we do not touch it.
        def inline(text: str) -> str:
            return re.sub(
                r"\s+", " ", text.replace("\r", " ").replace("\n", " ")
            ).strip()

        parts: List[str] = []
        parts.append(f"{cls.DIALOGUE_ID_LINE_PREFIX}{dialogue_id}\n")
        parts.append(f"{cls.SOURCE_LINE_PREFIX}{source}\n")
        parts.append(f"{cls.STORY_MARKER}\n")
        parts.append(story.rstrip("\n"))
        parts.append("\n")
        parts.append(f"{cls.CONVERSATION_MARKER}\n")

        for i, (q, a) in enumerate(history, start=1):
            parts.append(f"Q{i}: {inline(q)}\n")
            parts.append(f"A{i}: {inline(a)}\n")

        parts.append(f"Q{turn_number}: {inline(question)}\n")
        parts.append(f"A{turn_number}:")  # critical: no trailing newline

        return "".join(parts)


class CoqaPromptParser:
    """Parser for prompts produced by CoqaPromptFormatter."""

    _DIALOGUE_ID_RE = re.compile(r"^### COQA_DIALOGUE_ID: (.+)$", re.MULTILINE)
    _CONV_Q_RE = re.compile(r"^Q(\d+):", re.MULTILINE)

    @classmethod
    def parse(cls, prompt: str) -> ParsedCoqaPrompt:
        """
        Parse (dialogue_id, turn_number) from a formatted prompt.

        Raises
        ------
        ValueError
            If expected markers are missing.
        """
        m = cls._DIALOGUE_ID_RE.search(prompt)
        if not m:
            raise ValueError(
                "Prompt missing dialogue id line: '### COQA_DIALOGUE_ID: ...'"
            )

        dialogue_id = m.group(1).strip()

        marker = f"{CoqaPromptFormatter.CONVERSATION_MARKER}\n"
        pos = prompt.find(marker)
        if pos < 0:
            raise ValueError(
                "Prompt missing conversation marker line: '### CONVERSATION'"
            )

        conv_text = prompt[pos + len(marker) :]
        q_nums = [int(mm.group(1)) for mm in cls._CONV_Q_RE.finditer(conv_text)]
        if not q_nums:
            raise ValueError(
                "Prompt has no question lines 'Q{k}:' after '### CONVERSATION'"
            )

        turn_number = max(q_nums)
        return ParsedCoqaPrompt(dialogue_id=dialogue_id, turn_number=turn_number)
