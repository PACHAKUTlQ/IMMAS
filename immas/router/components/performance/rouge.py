"""
immas.router.components.performance.rouge

ROUGE evaluator.

- Extracts the assistant's "final answer" as the last non-empty line of the model output.
- Compares it to the dataset gold answer for (dialogue_id, turn_number) using ROUGE.
- Marks correct if configured ROUGE metric F1 >= threshold.

Normalization
-------------
Uses shared normalization utilities:
- strip "A3:" / "Final Answer:"
- numeric normalization (number words, digit commas)
- collapse whitespace
- optional lowercase
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from rouge import Rouge

from immas.data.coqa.loader import CoqaDatasetIndex
from immas.openai.chat import extract_first_assistant_message
from immas.router.components.performance.base import PerformanceEvalContext
from immas.router.components.performance.normalization import (
    extract_last_nonempty_line,
    normalize_answer_line,
)


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
    before normalization.
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
        # Backward-compatible wrapper used elsewhere in the router.
        return extract_last_nonempty_line(text)

    @classmethod
    def normalize_answer_line(cls, line: str, *, lowercase: bool) -> str:
        # Backward-compatible wrapper.
        return normalize_answer_line(line, lowercase=lowercase)

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

        hypothesis_last_line_raw = extract_last_nonempty_line(assistant.content)
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

        hypothesis = normalize_answer_line(
            hypothesis_last_line_raw, lowercase=bool(self.lowercase)
        )
        reference = normalize_answer_line(
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
        """

        res = self.score(ctx)
        return bool(res.correct) if res is not None else False
