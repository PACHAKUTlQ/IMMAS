"""
immas.router.components.performance

Performance evaluation hooks.

The router uses online learning and logs a `correct` field. This module provides
a single interface for evaluating correctness for both normal traffic and warmup.

This file now supports two evaluator modes:
- AlwaysCorrectEvaluator: placeholder (always True)
- RougeCoqaEvaluator: dataset-backed evaluation using CoQA gold answers and ROUGE

RougeCoqaEvaluator behavior
--------------------------
- Extracts the assistant's "final answer" as the last non-empty line of the model
  output (as instructed by the prompt).
- Compares it to the dataset gold answer for (dialogue_id, turn_number) using
  the `rouge` library (https://pypi.org/project/rouge/).
- Marks `correct=True` if the chosen ROUGE metric F1 >= threshold.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from rouge import Rouge

from immas.data.coqa.loader import CoqaDatasetIndex
from immas.openai.chat import extract_first_assistant_message


@dataclass(frozen=True, slots=True)
class PerformanceEvalContext:
    """Inputs used for evaluating completion performance/correctness."""

    run_id: str
    dialogue_id: str
    turn_number: int
    source: str
    request_body: Mapping[str, Any]
    response_json: Mapping[str, Any]


class PerformanceEvaluator(Protocol):
    """Protocol for evaluating correctness/performance of one completion."""

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:
        """
        Evaluate whether the completion should be considered correct.

        Notes
        -----
        This project may use dataset-based exact match, semantic match, LLM-as-a-judge,
        or other metrics. The router only needs a boolean to train a lightweight
        performance classifier and to log a stable signal.
        """
        ...


@dataclass(frozen=True, slots=True)
class AlwaysCorrectEvaluator:
    """
    Placeholder evaluator that always returns True.

    This keeps the router ready for a real evaluator implementation without
    requiring changes to warmup or request processing flow.
    """

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:  # noqa: ARG002
        return True


@dataclass(frozen=True, slots=True)
class RougeCoqaEvaluator:
    """
    CoQA performance evaluator using ROUGE between:
    - hypothesis: last non-empty line of assistant output (final answer)
    - reference: dataset gold answer for (dialogue_id, turn_number)

    Parameters
    ----------
    dataset
        Preloaded CoQA dataset index.
    rouge_metric
        One of "rouge-1", "rouge-2", "rouge-l".
    f1_threshold
        Mark correct if ROUGE(metric).f >= this threshold.
    lowercase
        If True, lowercase both hypothesis and reference before scoring.
    """

    dataset: CoqaDatasetIndex
    rouge_metric: str = "rouge-l"
    f1_threshold: float = 0.3
    lowercase: bool = True

    # We keep a single Rouge instance for efficiency.
    # In asyncio, evaluation runs on the event loop thread; this is safe in practice.
    _rouge: Rouge = Rouge()

    _WS_RE = re.compile(r"\s+")
    _A_TURN_PREFIX_RE = re.compile(r"^\s*A\s*\d+\s*:\s*", re.IGNORECASE)
    _FINAL_PREFIX_RE = re.compile(
        r"^\s*(?:final\s*answer|answer)\s*[:\-]\s*", re.IGNORECASE
    )

    def __post_init__(self) -> None:
        metric = str(self.rouge_metric).strip().lower()
        if metric not in {"rouge-1", "rouge-2", "rouge-l"}:
            raise ValueError(
                f"Unsupported rouge_metric={self.rouge_metric!r}. "
                "Expected one of: rouge-1, rouge-2, rouge-l"
            )
        if not (0.0 <= float(self.f1_threshold) <= 1.0):
            raise ValueError(f"f1_threshold must be in [0,1], got {self.f1_threshold}")

    @classmethod
    def _extract_last_nonempty_line(cls, text: str) -> str:
        """
        Extract the last non-empty line of a model output.

        If all lines are empty, returns the stripped full text.
        """

        s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = s.split("\n")
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return s.strip()

    @classmethod
    def _normalize_answer_line(cls, line: str, *, lowercase: bool) -> str:
        """
        Normalize a single-line answer for ROUGE:
        - strip common prefixes (e.g., "A3:", "Final Answer:")
        - collapse whitespace
        - optional lowercase
        """

        s = str(line or "").strip()
        s = cls._A_TURN_PREFIX_RE.sub("", s)
        s = cls._FINAL_PREFIX_RE.sub("", s)
        s = cls._WS_RE.sub(" ", s).strip()
        if lowercase:
            s = s.lower()
        return s

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:
        """
        Evaluate correctness for one completion.

        Returns False if:
        - the completion has no assistant message content,
        - turn_number is invalid,
        - the dataset lookup fails,
        - ROUGE scoring fails or produces missing fields,
        - or the score is below threshold.
        """

        if int(ctx.turn_number) < 1:
            return False

        assistant = extract_first_assistant_message(ctx.response_json)
        if assistant is None:
            return False

        hypothesis_raw = self._extract_last_nonempty_line(assistant.content)
        hypothesis = self._normalize_answer_line(
            hypothesis_raw, lowercase=self.lowercase
        )

        if not hypothesis:
            return False

        try:
            reference_raw = self.dataset.get_answer(
                dialogue_id=str(ctx.dialogue_id), turn_number=int(ctx.turn_number)
            )
        except Exception:
            return False

        reference = self._normalize_answer_line(reference_raw, lowercase=self.lowercase)
        if not reference:
            return False

        try:
            scores = self._rouge.get_scores(hypothesis, reference, avg=True)
        except Exception:
            return False

        metric = str(self.rouge_metric).strip().lower()
        m = scores.get(metric)
        if not isinstance(m, Mapping):
            return False

        f = m.get("f")
        try:
            f1 = float(f)
        except Exception:
            return False

        return bool(f1 >= float(self.f1_threshold))
