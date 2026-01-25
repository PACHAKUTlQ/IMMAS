"""
immas.router.components.performance.normalization

Shared normalization utilities for dataset-backed evaluators.

This module provides:
- last-line extraction (router prompts ask for "final answer" on last line),
- numeric normalization (e.g., "one hundred" -> "100", "1,234" -> "1234"),
- lightweight answer-prefix stripping (e.g. "A3:", "Final Answer:"),
- tokenization suitable for token-span matching.
"""

from __future__ import annotations

import re
import unicodedata

from typing import Any, Iterable, Mapping, cast

# NOTE: intentionally small + conventional. We do not try to be a full English
# number parser, but we *do* handle common answer forms that break scoring.
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

_WS_RE = re.compile(r"\s+")
_A_TURN_PREFIX_RE = re.compile(r"^\s*A\s*\d+\s*:\s*", re.IGNORECASE)
_FINAL_PREFIX_RE = re.compile(r"(?i)\b(final\s+)?answer\s*:\s*", re.IGNORECASE)

_ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})


def extract_last_nonempty_line(text: str) -> str:
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


def strip_common_answer_prefixes(text: str) -> str:
    """
    Remove common answer markers such as "A3:" or "Final Answer:".
    """

    s = str(text or "").strip()
    s = _A_TURN_PREFIX_RE.sub("", s)
    s = _FINAL_PREFIX_RE.sub("", s)

    return s.strip()


def normalize_numbers(text: str) -> str:
    """
    Normalize numeric expressions in `text` to improve scoring robustness.

    Transformations:
    1) Digit comma stripping: "1,234" -> "1234"
    2) Conservative number-word conversion: e.g. "one hundred and five" -> "105"
       and "three point five" -> "3.5" (decimal units only).
    """

    s = str(text or "")
    if not s:
        return ""

    s = _NUMERIC_COMMA_RE.sub("", s)
    s = _LETTER_HYPHEN_RE.sub(" ", s)

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
        if not words:
            return None

        cleaned: list[str] = []
        for i, w in enumerate(words):
            if w == "and":
                continue
            if w in {"a", "an"}:
                nxt = words[i + 1] if i + 1 < len(words) else ""
                if nxt in _SCALES:
                    cleaned.append("one")
                    continue
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

            break

        parsed = _try_parse_number_words(words)

        prev_word = _prev_alpha(start)
        next_word = _next_alpha(j)

        contains_scale_or_point = any(w in _SCALES or w == "point" for w in words)

        should_convert = False
        if parsed is not None:
            if numeric_dominant:
                should_convert = True
            elif contains_scale_or_point or len(words) >= 2:
                should_convert = True
            elif len(words) == 1:
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


def normalize_answer_line(text: str, *, lowercase: bool) -> str:
    """
    Normalize a single-line answer for deterministic scoring:

    - strip common prefixes (e.g., "A3:", "Final Answer:")
    - normalize numeric formatting (number-words + digit commas)
    - collapse whitespace
    - optional lowercase
    """

    s = strip_common_answer_prefixes(text)
    s = normalize_numbers(s)
    s = _WS_RE.sub(" ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def _is_wordish_category(cat: str) -> bool:
    # Letters, marks (accents), numbers.
    return bool(cat) and cat[0] in {"L", "M", "N"}


def tokenize_words_for_span(text: str) -> list[str]:
    """
    Tokenize text into "word tokens" for token-span matching.

    Behavior:
    - Unicode NFD normalization
    - sequences of letters/marks/numbers are grouped
    - decimal points are kept inside numeric tokens when surrounded by digits
    - punctuation and symbols are treated as separators (not emitted)
    """

    s = unicodedata.normalize("NFD", str(text or ""))
    if not s:
        return []

    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    n = len(s)
    for i, ch in enumerate(s):
        if ch.isspace():
            flush()
            continue

        cat = unicodedata.category(ch)

        if _is_wordish_category(cat):
            buf.append(ch)
            continue

        # Keep '.' inside numbers like "3.5"
        if ch == "." and buf and buf[-1].isdigit():
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt.isdigit():
                buf.append(ch)
                continue

        # Otherwise: separator (punct/symbol/etc).
        flush()

    flush()

    return tokens


def contains_token_span(
    *, haystack_tokens: Iterable[str], needle_tokens: Iterable[str]
) -> bool:
    """
    Return True if needle_tokens appears as a contiguous subsequence in haystack_tokens.
    """

    hay = list(haystack_tokens)
    ned = list(needle_tokens)

    if not ned:
        return False
    if len(ned) > len(hay):
        return False

    # Simple O(n*m) scan; sequences are very small in this use-case.
    m = len(ned)
    for i in range(len(hay) - m + 1):
        if hay[i : i + m] == ned:
            return True

    return False


def normalize_and_tokenize_for_span(
    *,
    text: str,
    lowercase: bool,
    drop_articles: bool = True,
) -> list[str]:
    """
    Normalize an answer line and tokenize it for token-span matching.

    Steps:
    - normalize_answer_line (prefix stripping + numeric normalization + whitespace + lowercase)
    - tokenize_words_for_span
    - optionally drop articles ("a", "an", "the")
    """

    s = normalize_answer_line(text, lowercase=lowercase)
    toks = tokenize_words_for_span(s)
    if not toks:
        return []

    if not drop_articles:
        return toks

    if lowercase:
        return [t for t in toks if t not in _ARTICLES]

    # If not lowercasing, do a conservative case-insensitive article drop.
    return [t for t in toks if t.lower() not in _ARTICLES]


def safe_get_mapping_str(m: Mapping[str, Any], key: str) -> str:
    """
    Internal helper for defensive extraction of strings from JSON-like dicts.
    """

    v = m.get(key)

    return str(cast(Any, v) or "")
