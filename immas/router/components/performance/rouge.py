"""
immas.router.components.performance.rouge

ROUGE evaluator (now supports multi-reference datasets like QuAC).

- Extracts the assistant's "final answer" as the last non-empty line of the model output.
- Looks up gold reference(s) for (dialogue_id, turn_number):
  - If dataset has get_answers(...)->list[str], use those (multi-ref).
  - Else fall back to get_answer(...)->str (single-ref).
- Applies QuAC-style handle_cannot to refs:
  - If #CANNOTANSWER >= #non-cannot spans => refs = ["CANNOTANSWER"]
  - Else drop CANNOTANSWER refs.
- Computes ROUGE between hypothesis and each reference; uses the BEST score.
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
from typing import Any, List, Mapping, cast

from rouge import Rouge

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

    `reference_raw` is the single reference that achieved the best score
    (for multi-reference datasets).
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
    Performance evaluator using ROUGE between:
    - hypothesis: last non-empty line of assistant output (final answer)
    - reference(s): dataset gold answer(s) for (dialogue_id, turn_number)

    Parameters
    ----------
    dataset
        Dataset index object. Must implement either:
        - get_answer(dialogue_id, turn_number) -> str
        - OR get_answers(dialogue_id, turn_number) -> list[str]
    rouge_metric
        One of "rouge-1", "rouge-2", "rouge-l".
    f1_threshold
        Mark correct if ROUGE(metric).f >= this threshold.
    lowercase
        If True, lowercase both hypothesis and reference before scoring.
    """

    dataset: Any
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
        return extract_last_nonempty_line(text)

    @classmethod
    def normalize_answer_line(cls, line: str, *, lowercase: bool) -> str:
        return normalize_answer_line(line, lowercase=lowercase)

    @staticmethod
    def _f1_from_metric(scores: Mapping[str, Any], metric: str) -> float | None:
        m = scores.get(metric)
        if not isinstance(m, Mapping):
            return None
        f = m.get("f")
        try:
            return float(cast(Any, f))
        except Exception:
            return None

    @staticmethod
    def _handle_cannot(refs: List[str]) -> List[str]:
        cleaned = [str(r) for r in refs if str(r or "").strip()]
        num_cannot = sum(1 for r in cleaned if r.strip() == "CANNOTANSWER")
        num_spans = sum(1 for r in cleaned if r.strip() != "CANNOTANSWER")
        if num_cannot >= num_spans:
            return ["CANNOTANSWER"]
        return [r for r in cleaned if r.strip() != "CANNOTANSWER"]

    def _get_references_raw(
        self, *, dialogue_id: str, turn_number: int
    ) -> List[str] | None:
        try:
            if hasattr(self.dataset, "get_answers"):
                refs_any = self.dataset.get_answers(
                    dialogue_id=str(dialogue_id), turn_number=int(turn_number)
                )
                if isinstance(refs_any, list):
                    refs = [str(x) for x in refs_any]
                else:
                    refs = [str(refs_any)]
            else:
                ref = self.dataset.get_answer(
                    dialogue_id=str(dialogue_id), turn_number=int(turn_number)
                )
                refs = [str(ref)]
        except Exception:
            return None

        refs = [r for r in refs if str(r or "").strip()]
        refs = self._handle_cannot(refs)
        return refs or None

    def score(self, ctx: PerformanceEvalContext) -> RougeScoreResult | None:
        if int(ctx.turn_number) < 1:
            return None

        assistant = extract_first_assistant_message(ctx.response_json)
        if assistant is None:
            return None

        hypothesis_last_line_raw = extract_last_nonempty_line(assistant.content)
        if not hypothesis_last_line_raw.strip():
            return None

        refs_raw = self._get_references_raw(
            dialogue_id=str(ctx.dialogue_id), turn_number=int(ctx.turn_number)
        )
        if not refs_raw:
            return None

        hypothesis = normalize_answer_line(
            hypothesis_last_line_raw, lowercase=bool(self.lowercase)
        )
        if not hypothesis:
            return None

        metric_used = str(self.rouge_metric).strip().lower()

        best_used_f1 = -1.0
        best_ref_raw: str | None = None
        best_f1: RougeF1Triple | None = None

        for reference_raw in refs_raw:
            if not str(reference_raw or "").strip():
                continue

            reference = normalize_answer_line(
                str(reference_raw), lowercase=bool(self.lowercase)
            )
            if not reference:
                continue

            try:
                scores = self._rouge.get_scores(hypothesis, reference, avg=True)
            except Exception:
                continue

            r1 = self._f1_from_metric(scores, "rouge-1")
            r2 = self._f1_from_metric(scores, "rouge-2")
            rl = self._f1_from_metric(scores, "rouge-l")
            if r1 is None or r2 is None or rl is None:
                continue

            f1 = RougeF1Triple(
                rouge_1_f1=float(r1), rouge_2_f1=float(r2), rouge_l_f1=float(rl)
            )
            used_f1 = float(f1.used_f1(metric_used))

            if used_f1 > best_used_f1:
                best_used_f1 = used_f1
                best_ref_raw = str(reference_raw)
                best_f1 = f1

        if best_ref_raw is None or best_f1 is None or best_used_f1 < 0.0:
            return None

        correct = bool(best_used_f1 >= float(self.f1_threshold))

        return RougeScoreResult(
            hypothesis_last_line_raw=str(hypothesis_last_line_raw),
            reference_raw=str(best_ref_raw),
            f1=best_f1,
            metric_used=str(metric_used),
            used_f1=float(best_used_f1),
            correct=bool(correct),
        )

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:
        res = self.score(ctx)
        return bool(res.correct) if res is not None else False
