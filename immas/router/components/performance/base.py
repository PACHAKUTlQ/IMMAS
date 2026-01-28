"""
immas.router.components.performance.base

Base types for router performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class PerformanceEvalContext:
    """
    Inputs used for evaluating completion performance/correctness.

    Notes
    -----
    This is intentionally minimal and stable: evaluators should treat it as
    immutable input, and any model-specific parsing should be done internally.
    """

    run_id: str
    dialogue_id: str
    turn_number: int
    source: str
    request_body: Mapping[str, Any]
    response_json: Mapping[str, Any]


class PerformanceEvaluator(Protocol):
    """
    Protocol for evaluating correctness/performance of one completion.

    Evaluators must be synchronous and non-blocking from the router's perspective
    (no network calls, no LLM calls).
    """

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

    Used when performance evaluation is disabled or unavailable.
    """

    def evaluate(self, ctx: PerformanceEvalContext) -> bool:  # noqa: ARG002
        return True
