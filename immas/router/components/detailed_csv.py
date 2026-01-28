"""
immas.router.components.detailed_csv

Clean CSV logger for inspecting dataset-level fields and ROUGE scores.

This is meant for validating:
- story/question/gold answer
- model output (full) and the last-line answer actually scored
- ROUGE F1 breakdown (rouge-1 / rouge-2 / rouge-l) and the metric actually used

The logger is asynchronous: a single background writer task consumes queued rows.
"""

from __future__ import annotations

import asyncio
import csv

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _sanitize_text_field(x: str) -> str:
    """
    Sanitize free-form text for CSV:

    - normalize newlines to \\n (literal two chars) to keep one record per line
    - keep other characters as-is (csv module will quote as needed)
    """

    s = str(x or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")


def _fmt_f(x: float | None) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""


@dataclass(frozen=True, slots=True)
class RouterDetailedCsvRow:
    """
    A single clean CSV row.

    Column order is defined by `FIELDNAMES` below.
    """

    source: str
    dialogue_id: str
    turn_number: int

    story: str
    question: str
    gold_answer: str

    llm_answer_last_line: str
    llm_answer: str

    evaluator: str
    correct: bool

    # ROUGE-specific
    rouge_metric_used: Optional[str] = None
    rouge_used_f1: Optional[float] = None
    rouge_1_f1: Optional[float] = None
    rouge_2_f1: Optional[float] = None
    rouge_l_f1: Optional[float] = None

    # Token-span-specific
    token_span_matched: Optional[bool] = None

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to a CSV-ready mapping of strings.

        We sanitize large text fields to remain single-line CSV cells.
        """

        d: Dict[str, Any] = asdict(self)
        return {
            "source": str(d["source"]),
            "dialogue_id": str(d["dialogue_id"]),
            "turn_number": str(int(d["turn_number"])),
            "story": _sanitize_text_field(str(d["story"])),
            "question": _sanitize_text_field(str(d["question"])),
            "gold_answer": _sanitize_text_field(str(d["gold_answer"])),
            "llm_answer_last_line": _sanitize_text_field(
                str(d["llm_answer_last_line"])
            ),
            "llm_answer": _sanitize_text_field(str(d["llm_answer"])),
            "evaluator": str(d["evaluator"]),
            "correct": str(bool(d["correct"])),
            "token_span_matched": str(d.get("token_span_matched") or ""),
            "rouge_metric_used": str(d.get("rouge_metric_used") or ""),
            "rouge_used_f1": _fmt_f(d.get("rouge_used_f1")),
            "rouge_1_f1": _fmt_f(d.get("rouge_1_f1")),
            "rouge_2_f1": _fmt_f(d.get("rouge_2_f1")),
            "rouge_l_f1": _fmt_f(d.get("rouge_l_f1")),
        }


FIELDNAMES: tuple[str, ...] = (
    "source",
    "dialogue_id",
    "turn_number",
    "story",
    "question",
    "gold_answer",
    "llm_answer_last_line",
    "llm_answer",
    "evaluator",
    "correct",
    "token_span_matched",
    "rouge_metric_used",
    "rouge_used_f1",
    "rouge_1_f1",
    "rouge_2_f1",
    "rouge_l_f1",
)


class AsyncDetailedCsvLogger:
    """
    Async CSV logger with a single background writer task.

    Notes
    -----
    - If `append=True`, the header is only written if the file is empty/non-existent.
    - Logging failures should not impact request handling; callers should treat
      `.log(...)` as best-effort.
    """

    def __init__(
        self, path: str, *, append: bool = False, flush_every: int = 1
    ) -> None:
        self._path = Path(path)
        self._append = bool(append)
        self._flush_every = int(flush_every)
        if self._flush_every < 1:
            raise ValueError("flush_every must be >= 1")

        self._q: "asyncio.Queue[Optional[Dict[str, str]]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    async def __aenter__(self) -> "AsyncDetailedCsvLogger":
        self._task = asyncio.create_task(self._writer_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def log(self, row: RouterDetailedCsvRow) -> None:
        """Enqueue one row for background writing."""

        await self._q.put(row.to_dict())

    async def close(self) -> None:
        """Flush and stop writer task (idempotent)."""

        if self._task is None:
            return
        await self._q.put(None)
        await self._task
        self._task = None

    def _should_write_header(self) -> bool:
        if not self._append:
            return True
        try:
            return (not self._path.exists()) or (self._path.stat().st_size <= 0)
        except Exception:
            return True

    async def _writer_loop(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        n = 0

        with self._path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(FIELDNAMES),
                extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,
            )
            if self._should_write_header():
                writer.writeheader()
                f.flush()

            while True:
                item = await self._q.get()
                if item is None:
                    break
                writer.writerow(item)
                n += 1
                if n % self._flush_every == 0:
                    f.flush()
            f.flush()
