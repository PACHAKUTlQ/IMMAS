"""
immas.router.components.performance.token_span

TokenSpan evaluator.

This is a fast, deterministic evaluator intended for real-time routing:
- Extracts assistant final answer as last non-empty line.
- Applies the same prefix stripping + numeric normalization pipeline.
- Tokenizes into word/number tokens (punctuation treated as separators).
- Returns True iff the gold answer token sequence appears as a contiguous span
  within the predicted answer token sequence.

No LLM fallback is used (by design).
"""

from __future__ import annotations

from dataclasses import dataclass

from immas.data.coqa.loader import CoqaDatasetIndex
from immas.openai.chat import extract_first_assistant_message
from immas.router.components.performance.base import PerformanceEvalContext
from immas.router.components.performance.normalization import (
    contains_token_span,
    extract_last_nonempty_line,
    normalize_and_tokenize_for_span,
)


@dataclass(frozen=True, slots=True)
class TokenSpanScoreResult:
    """
    Detailed token-span scoring result.

    This is primarily useful for debugging/analysis and is kept lightweight.
    """

    hypothesis_last_line_raw: str
    reference_raw: str
    hypothesis_tokens: tuple[str, ...]
    reference_tokens: tuple[str, ...]
    matched: bool


@dataclass(frozen=True, slots=True)
class TokenSpanCoqaEvaluator:
    """
    Token-span evaluator against CoQA gold answers.

    Parameters
    ----------
    dataset
        Preloaded CoQA dataset index.
    lowercase
        If True, lowercase both hypothesis and reference during normalization.
    """

    dataset: CoqaDatasetIndex
    lowercase: bool = True

    def score(self, ctx: PerformanceEvalContext) -> TokenSpanScoreResult | None:
        """
        Compute a token-span match result for one completion.

        Returns None if:
        - invalid turn_number
        - no assistant output
        - dataset lookup fails
        - hypothesis/reference is empty after normalization/tokenization
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

        hyp_tokens = normalize_and_tokenize_for_span(
            text=hypothesis_last_line_raw,
            lowercase=bool(self.lowercase),
            drop_articles=True,
        )
        ref_tokens = normalize_and_tokenize_for_span(
            text=str(reference_raw),
            lowercase=bool(self.lowercase),
            drop_articles=True,
        )

        if not hyp_tokens or not ref_tokens:
            return None

        matched = contains_token_span(
            haystack_tokens=hyp_tokens,
            needle_tokens=ref_tokens,
        )

        return TokenSpanScoreResult(
            hypothesis_last_line_raw=str(hypothesis_last_line_raw),
            reference_raw=str(reference_raw),
            hypothesis_tokens=tuple(hyp_tokens),
            reference_tokens=tuple(ref_tokens),
            matched=bool(matched),
        )

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:
        """
        Return True iff token-span matching succeeds; otherwise False.

        Notes
        -----
        This evaluator is intentionally strict and non-semantic:
        it never calls an LLM and never performs embedding similarity, etc.
        """

        res = self.score(ctx)
        return bool(res.matched) if res is not None else False
