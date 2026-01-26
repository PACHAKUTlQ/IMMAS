"""
immas.router.components.performance

Performance evaluation package.

Exports the public evaluator interfaces and implementations used by the router.
"""

from __future__ import annotations

from immas.router.components.performance.base import (
    AlwaysCorrectEvaluator,
    PerformanceEvalContext,
    PerformanceEvaluator,
)
from immas.router.components.performance.rouge import (
    RougeCoqaEvaluator,
    RougeF1Triple,
    RougeScoreResult,
)
from immas.router.components.performance.token_span import (
    TokenSpanCoqaEvaluator,
    TokenSpanScoreResult,
)

__all__ = [
    "AlwaysCorrectEvaluator",
    "PerformanceEvalContext",
    "PerformanceEvaluator",
    "RougeCoqaEvaluator",
    "RougeF1Triple",
    "RougeScoreResult",
    "TokenSpanCoqaEvaluator",
    "TokenSpanScoreResult",
]
