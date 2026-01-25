"""
immas.router.components.performance

Performance evaluation hooks.

The router uses online learning and logs a `correct` field. This module provides
a single interface for evaluating correctness for both normal traffic and warmup.

This file supports:
- AlwaysCorrectEvaluator: placeholder (always True)
- RougeCoqaEvaluator: dataset-backed evaluation using CoQA gold answers and ROUGE

RougeCoqaEvaluator behavior
--------------------------
- Extracts the assistant's "final answer" as the last non-empty line of the model
  output (as instructed by the prompt).
- Compares it to the dataset gold answer for (dialogue_id, turn_number) using
  the `rouge` library (https://pypi.org/project/rouge/).
- Marks `correct=True` if the configured ROUGE metric F1 >= threshold.

Additionally, RougeCoqaEvaluator exposes `score(ctx)` returning:
- the raw gold answer and last-line hypothesis used for scoring
- ROUGE-1/2/L F1 scores
- the used metric and used F1
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, cast

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

        Returns a boolean signal suitable for online learning and logging.
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
class RougeF1Triple:
    """ROUGE F1 breakdown."""

    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float

    def used_f1(self, metric: str) -> float:
        m = str(metric).strip().lower()
        if m == "rouge-1":
            return float(self.rouge_1_f1)
        if m == "rouge-2":
            return float(self.rouge_2_f1)
        return float(self.rouge_l_f1)


@dataclass(frozen=True, slots=True)
class RougeScoreResult:
    """
    Detailed scoring result for one completion.

    `hypothesis_last_line_raw` is the last non-empty line of the assistant output,
    before normalization (prefix stripping, whitespace collapse, optional lowercase).
    """

    hypothesis_last_line_raw: str
    reference_raw: str
    f1: RougeF1Triple
    metric_used: str
    used_f1: float
    correct: bool


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

    _rouge: Rouge = field(default_factory=Rouge, init=False, repr=False, compare=False)

    _WS_RE = re.compile(r"\s+")
    _A_TURN_PREFIX_RE = re.compile(r"^\s*A\s*\d+\s*:\s*", re.IGNORECASE)
    _FINAL_PREFIX_RE = re.compile(r"(?i)\b(final\s+)?answer\s*:\s*", re.IGNORECASE)

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
    def extract_last_nonempty_line(cls, text: str) -> str:
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
    def normalize_answer_line(cls, line: str, *, lowercase: bool) -> str:
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

    @staticmethod
    def _f1_from_metric(scores: Mapping[str, Any], metric: str) -> float | None:
        """
        Extract F1 from rouge.get_scores(..., avg=True) output for a metric key.
        """

        m = scores.get(metric)
        if not isinstance(m, Mapping):
            return None
        f = m.get("f")
        try:
            return float(cast(Any, f))
        except Exception:
            return None

    def score(self, ctx: PerformanceEvalContext) -> RougeScoreResult | None:
        """
        Compute detailed ROUGE scoring result for one completion.

        Returns None if:
        - invalid turn_number
        - no assistant output
        - dataset lookup fails
        - ROUGE scoring fails
        """

        if int(ctx.turn_number) < 1:
            return None

        assistant = extract_first_assistant_message(ctx.response_json)
        if assistant is None:
            return None

        hypothesis_last_line_raw = self.extract_last_nonempty_line(assistant.content)
        if not hypothesis_last_line_raw.strip():
            return None

        try:
            reference_raw = self.dataset.get_answer(
                dialogue_id=str(ctx.dialogue_id), turn_number=int(ctx.turn_number)
            )
        except Exception:
            return None

        if not str(reference_raw or "").strip():
            return None

        hypothesis = self.normalize_answer_line(
            hypothesis_last_line_raw, lowercase=bool(self.lowercase)
        )
        reference = self.normalize_answer_line(
            str(reference_raw), lowercase=bool(self.lowercase)
        )
        if not hypothesis or not reference:
            return None

        try:
            scores = self._rouge.get_scores(hypothesis, reference, avg=True)
        except Exception:
            return None

        r1 = self._f1_from_metric(scores, "rouge-1")
        r2 = self._f1_from_metric(scores, "rouge-2")
        rl = self._f1_from_metric(scores, "rouge-l")
        if r1 is None or r2 is None or rl is None:
            return None

        f1 = RougeF1Triple(
            rouge_1_f1=float(r1), rouge_2_f1=float(r2), rouge_l_f1=float(rl)
        )

        metric_used = str(self.rouge_metric).strip().lower()
        used_f1 = float(f1.used_f1(metric_used))
        correct = bool(used_f1 >= float(self.f1_threshold))

        return RougeScoreResult(
            hypothesis_last_line_raw=str(hypothesis_last_line_raw),
            reference_raw=str(reference_raw),
            f1=f1,
            metric_used=str(metric_used),
            used_f1=float(used_f1),
            correct=bool(correct),
        )

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:
        """
        Evaluate correctness for one completion.

        Equivalent to `score(ctx)` followed by thresholding on the configured metric.
        """

        res = self.score(ctx)
        return bool(res.correct) if res is not None else False
