"""
immas.router.predictor

Online predictor for latency/cost/performance plus a cache-reuse calibrator.

Important:
- We never use observed cached_tokens as a prediction-time feature.
- We *do* train a cache-ratio model using observed cached_tokens as labels.
- Latency/cost models receive the *predicted* cache ratio as an input feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from river import compose, preprocessing, tree


Features = Dict[str, Any]
MetricPred = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class PredictorInput:
    """Inputs known at routing time."""

    model: str
    source: str
    dialogue_id: str
    turn_number: int

    # Deterministic serialization of messages (or other stable representation)
    # Used only via derived scalar features (length).
    prompt_repr: str

    # Router-computed proxy for cache reuse (prefix match ratio in [0,1])
    kvmatch_text: float

    router_inflight: int = 0
    router_rps_1s: float = 0.0


class AgentPredictor:
    """
    Online predictor for:
    - latency_ms (router E2E for now)
    - cost_tokens (total tokens)
    - performance_prob (placeholder)
    - cache_ratio (calibrator target: cached_tokens / prompt_tokens)
    """

    def __init__(self) -> None:
        enc = preprocessing.OneHotEncoder()

        self.model_cache_ratio = compose.Pipeline(
            enc,
            tree.HoeffdingTreeRegressor(),
        )
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

    def _base_features(self, inp: PredictorInput) -> Features:
        return {
            "bias": 1.0,
            "model": inp.model,
            "source": inp.source,
            "turn_number": float(inp.turn_number),
            "prompt_chars": float(len(inp.prompt_repr)),
            "kvmatch_text": float(inp.kvmatch_text),
            "router_inflight": float(inp.router_inflight),
            "router_rps_1s": float(inp.router_rps_1s),
        }

    def _predict_cache_ratio(self, x_base: Features) -> float:
        v = self.model_cache_ratio.predict_one(x_base)
        r = float(v) if v is not None else 0.0
        return max(0.0, min(1.0, r))

    def _full_features(self, inp: PredictorInput) -> Features:
        x_base = self._base_features(inp)
        pred_cache_ratio = self._predict_cache_ratio(x_base)
        return {**x_base, "pred_cache_ratio": float(pred_cache_ratio)}

    def predict(self, inp: PredictorInput) -> Dict[str, MetricPred]:
        x_base = self._base_features(inp)
        pred_cache_ratio = self._predict_cache_ratio(x_base)
        x = {**x_base, "pred_cache_ratio": float(pred_cache_ratio)}

        lat = self.model_latency.predict_one(x)
        cost = self.model_cost.predict_one(x)

        lat_f = max(0.0, float(lat) if lat is not None else 0.0)
        cost_f = max(0.0, float(cost) if cost is not None else 0.0)

        proba = self.model_perf.predict_proba_one(x)
        perf_f = float(proba.get(True, 0.0)) if isinstance(proba, Mapping) else 0.0
        perf_f = max(0.0, min(1.0, perf_f))

        dummy_std = 0.0
        return {
            "latency_ms": (lat_f, dummy_std),
            "cost_tokens": (cost_f, dummy_std),
            "performance": (perf_f, dummy_std),
            "cache_ratio": (float(pred_cache_ratio), dummy_std),
        }

    def update(
        self,
        inp: PredictorInput,
        *,
        real_latency_ms: float,
        real_cost_tokens: int,
        real_perf_correct: bool,
        real_cache_ratio: float | None,
    ) -> None:
        # Use cache ratio prediction as an input to other models (no leakage).
        x_base = self._base_features(inp)
        pred_cache_ratio = self._predict_cache_ratio(x_base)
        x = {**x_base, "pred_cache_ratio": float(pred_cache_ratio)}

        self.model_latency.learn_one(x, float(real_latency_ms))
        self.model_cost.learn_one(x, float(real_cost_tokens))
        self.model_perf.learn_one(x, bool(real_perf_correct))

        # Train cache calibrator from router-known features only.
        if real_cache_ratio is not None:
            y = max(0.0, min(1.0, float(real_cache_ratio)))
            self.model_cache_ratio.learn_one(x_base, y)
