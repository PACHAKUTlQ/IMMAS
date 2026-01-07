"""
Online predictor for latency/cost/performance.

This is intentionally small but structured:
- Feature construction is centralized
- Uses River pipelines with OneHotEncoder to avoid manual one-hot bookkeeping
- Outputs keep the "value + dummy_std" shape for compatibility with UCB-style code
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from river import compose, preprocessing, tree


Features = Dict[str, Any]
MetricPred = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class PredictorInput:
    """Inputs known at routing time (basic version)."""

    model: str
    source: str
    dialogue_id: str
    turn_number: int
    prompt_text: str
    kvmatch: float


class AgentPredictor:
    """
    Online predictor for (latency_ms, cost_tokens, performance_prob).

    Notes
    -----
    - Latency is measured end-to-end wall time in this basic version.
    - Cost tokens are a crude proxy from the fake API (char-based).
    - Performance is correctness vs gold; with the fake API it's mainly a sanity check.
    """

    def __init__(self) -> None:
        # OneHotEncoder handles categorical string features automatically.
        self.model_latency = compose.Pipeline(
            preprocessing.OneHotEncoder(),
            tree.HoeffdingTreeRegressor(),
        )
        self.model_cost = compose.Pipeline(
            preprocessing.OneHotEncoder(),
            tree.HoeffdingTreeRegressor(),
        )
        self.model_perf = compose.Pipeline(
            preprocessing.OneHotEncoder(),
            tree.HoeffdingTreeClassifier(),
        )

    def make_features(self, inp: PredictorInput) -> Features:
        """
        Build a feature dict.

        This is where you will progressively add:
        - queue/backlog
        - model-side KV cache state (eviction pressure)
        - output-length estimates
        - etc.
        """
        return {
            "bias": 1.0,
            "model": inp.model,
            "source": inp.source,
            "turn_number": float(inp.turn_number),
            "prompt_chars": float(len(inp.prompt_text)),
            "kvmatch": float(inp.kvmatch),
        }

    def predict(self, inp: PredictorInput) -> Dict[str, MetricPred]:
        """
        Predict metrics.

        Returns
        -------
        dict
            {
              "latency_ms": (value, std_placeholder),
              "cost_tokens": (value, std_placeholder),
              "performance": (prob_correct, std_placeholder),
            }
        """
        x = self.make_features(inp)

        lat = self.model_latency.predict_one(x)
        cost = self.model_cost.predict_one(x)

        lat_f = float(lat) if lat is not None else 0.0
        cost_f = float(cost) if cost is not None else 0.0

        proba = self.model_perf.predict_proba_one(x)
        perf_f = float(proba.get(True, 0.0)) if isinstance(proba, Mapping) else 0.0

        # Domain constraints
        lat_f = max(0.0, lat_f)
        cost_f = max(0.0, cost_f)
        perf_f = max(0.0, min(1.0, perf_f))

        dummy_std = 0.0
        return {
            "latency_ms": (lat_f, dummy_std),
            "cost_tokens": (cost_f, dummy_std),
            "performance": (perf_f, dummy_std),
        }

    def update(
        self,
        inp: PredictorInput,
        *,
        real_latency_ms: float,
        real_cost_tokens: int,
        real_perf_correct: bool,
    ) -> None:
        """Online update with one observed datapoint."""
        x = self.make_features(inp)
        self.model_latency.learn_one(x, float(real_latency_ms))
        self.model_cost.learn_one(x, float(real_cost_tokens))
        self.model_perf.learn_one(x, bool(real_perf_correct))
