"""
immas.router.components.performance

Performance evaluation hooks.

The router uses online learning and logs a `correct` field. This module provides
a single interface for evaluating correctness for both normal traffic and warmup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


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
