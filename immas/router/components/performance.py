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

Robustness tweaks
-----------------
Before ROUGE scoring, both hypothesis and reference are normalized:
- common answer prefixes removed (e.g., "A3:", "Final Answer:")
- collapse whitespace
- optional lowercase
- remove commas inside digit sequences (e.g., "1,234" -> "1234")
- *light* number-word to numeral normalization for common cases (e.g., "eight" -> "8",
  "a hundred" -> "100"). This is intentionally conservative to avoid converting
  normal prose like "one of the ...".
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, cast

from rouge import Rouge

from immas.data.coqa.loader import CoqaDatasetIndex
from immas.openai.chat import extract_first_assistant_message


# NOTE: intentionally small + conventional. We do not try to be a full English
# number parser, but we *do* handle the common answer forms that break ROUGE.
_NUMBER_WORD_INT: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_NUM_WORDS: frozenset[str] = frozenset(set(_NUMBER_WORD_INT.keys()) | {"point"})
_NUM_JOINERS: frozenset[str] = frozenset({"and", "a", "an"})
_NUM_PHRASE_WORDS: frozenset[str] = frozenset(set(_NUM_WORDS) | set(_NUM_JOINERS))
_SCALES: frozenset[str] = frozenset({"hundred", "thousand", "million", "billion"})
_UNITS: frozenset[str] = frozenset(
    {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)
_TENS_TEENS: frozenset[str] = frozenset(
    {
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    }
)

# Used for tokenizing while preserving punctuation and whitespace so we can rebuild
# the string with minimal distortion.
_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|\s+|[^\w\s]", re.UNICODE)

# Remove comma separators inside digit sequences: "1,234" -> "1234"
_NUMERIC_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

# Convert letter-letter hyphen into space so "twenty-one" can be parsed.
_LETTER_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])[-–](?=[A-Za-z])")


def normalize_numbers(text: str) -> str:
    """
    Normalize numeric expressions in `text` to improve ROUGE robustness.

    This performs two transformations:

    1) Digit comma stripping
       - "1,234" -> "1234"

    2) Conservative number-word conversion
       - converts *common* standalone number answers and scale phrases:
         "eight" -> "8"
         "a hundred" -> "100"
         "one hundred" -> "100"
         "two thousand" -> "2000"
         "forty nine" -> "49"
         "one hundred and five" -> "105"
         "three point five" -> "3.5"  (decimal part supports unit words only)

    Safety heuristic (important):
    - If the string contains non-numeric prose (e.g., "one of the ..."), we avoid
      converting single-token number words like "one". Scale phrases like
      "a hundred" are still converted inside prose.

    Parameters
    ----------
    text
        Input string.

    Returns
    -------
    str
        Output string with numeric normalization applied.
    """

    s = str(text or "")
    if not s:
        return ""

    # Normalize digit formatting first.
    s = _NUMERIC_COMMA_RE.sub("", s)

    # Help parse hyphenated number words (e.g., "twenty-one").
    s = _LETTER_HYPHEN_RE.sub(" ", s)

    # Determine whether this looks like a "numeric-only" answer
    # If yes, we can be more aggressive converting single-token words.
    alpha_words = [w.lower() for w in re.findall(r"[A-Za-z]+", s)]
    numeric_dominant = bool(alpha_words) and all(
        w in _NUM_PHRASE_WORDS for w in alpha_words
    )

    tokens = _TOKEN_RE.findall(s)
    out: list[str] = []

    def _prev_alpha(idx: int) -> str | None:
        for k in range(idx - 1, -1, -1):
            t = tokens[k]
            if t.isalpha():
                return t.lower()
        return None

    def _next_alpha(idx: int) -> str | None:
        for k in range(idx, len(tokens)):
            t = tokens[k]
            if t.isalpha():
                return t.lower()
        return None

    def _try_parse_number_words(words: list[str]) -> str | None:
        """
        Attempt to parse a list of lowercase number-phrase words into a numeral string.

        Returns None if parsing is not possible / not confident.
        """

        if not words:
            return None

        # Remove joiner "and". Handle "a/an" only when it clearly stands for "one"
        # before a scale word (e.g. "a hundred", "an million" (rare)).
        cleaned: list[str] = []
        for i, w in enumerate(words):
            if w == "and":
                continue
            if w in {"a", "an"}:
                nxt = words[i + 1] if i + 1 < len(words) else ""
                if nxt in _SCALES:
                    cleaned.append("one")
                    continue
                # If it's just "a" (or "an") not tied to a numeric scale, bail.
                return None
            cleaned.append(w)

        if not cleaned:
            return None

        total = 0
        current = 0

        saw_point = False
        decimal_digits: list[str] = []

        def _flush_scale(scale: int) -> None:
            nonlocal total, current
            if current == 0:
                current = 1
            total += current * scale
            current = 0

        i = 0
        while i < len(cleaned):
            w = cleaned[i]

            if w == "point":
                # Decimal portion: only accept unit words as digits.
                if saw_point:
                    return None
                saw_point = True
                i += 1
                continue

            if saw_point:
                if w not in _UNITS:
                    return None
                decimal_digits.append(str(_NUMBER_WORD_INT[w]))
                i += 1
                continue

            if w in _UNITS or w in _TENS_TEENS:
                current += int(_NUMBER_WORD_INT[w])
                i += 1
                continue

            if w == "hundred":
                if current == 0:
                    current = 1
                current *= 100
                i += 1
                continue

            if w in {"thousand", "million", "billion"}:
                _flush_scale(int(_NUMBER_WORD_INT[w]))
                i += 1
                continue

            # Unknown word in phrase.
            return None

        value = total + current
        if saw_point:
            if not decimal_digits:
                return None
            return f"{int(value)}.{''.join(decimal_digits)}"
        return str(int(value))

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.isalpha():
            out.append(t)
            i += 1
            continue

        w0 = t.lower()
        if w0 not in _NUM_PHRASE_WORDS:
            out.append(t)
            i += 1
            continue

        # Consume a contiguous numeric word phrase, but *do not* consume trailing
        # whitespace unless the phrase continues after it.
        start = i
        words: list[str] = []
        j = i

        while j < len(tokens):
            tj = tokens[j]

            if tj.isalpha():
                wj = tj.lower()
                if wj in _NUM_PHRASE_WORDS:
                    words.append(wj)
                    j += 1
                    continue
                break

            # Only include whitespace/hyphen if the phrase continues with another
            # numeric phrase word afterwards.
            if tj.isspace() or tj in {"-", "–"}:
                k = j + 1
                while k < len(tokens) and (
                    tokens[k].isspace() or tokens[k] in {"-", "–"}
                ):
                    k += 1
                if (
                    k < len(tokens)
                    and tokens[k].isalpha()
                    and tokens[k].lower() in _NUM_PHRASE_WORDS
                ):
                    j += 1
                    continue
                break

            # Any other punctuation breaks the phrase.
            break

        # j is the first token not in the phrase (or trailing whitespace).
        parsed = _try_parse_number_words(words)

        prev_word = _prev_alpha(start)
        next_word = _next_alpha(j)

        contains_scale_or_point = any(w in _SCALES or w == "point" for w in words)

        # Conservative conversion policy:
        # - always convert numeric-only answers
        # - convert scale phrases and multi-word phrases
        # - avoid converting lone "one"/"two"/... inside prose ("one of ...")
        should_convert = False
        if parsed is not None:
            if numeric_dominant:
                should_convert = True
            elif contains_scale_or_point or len(words) >= 2:
                should_convert = True
            elif len(words) == 1:
                # Single-token conversions only in safe local contexts.
                # Specifically avoid "one of ..." and "... of one ..." patterns.
                w_single = words[0]
                if w_single in _UNITS or w_single in _TENS_TEENS:
                    if next_word != "of" and prev_word != "of":
                        should_convert = True

        if should_convert and parsed is not None:
            out.append(parsed)
        else:
            out.extend(tokens[start:j])

        i = j

    return "".join(out)


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
    before normalization (prefix stripping, numeric normalization, whitespace collapse,
    optional lowercase).
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
        - normalize numeric formatting (number-words + digit commas)
        - collapse whitespace
        - optional lowercase
        """

        s = str(line or "").strip()
        s = cls._A_TURN_PREFIX_RE.sub("", s)
        s = cls._FINAL_PREFIX_RE.sub("", s)

        s = normalize_numbers(s)

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
