"""
immas.router.predictor

Online predictor for latency/cost/performance.

This module now supports *multiple independent predictors*, one per backend, to
avoid cross-backend interference during online learning.

Key idea
--------
Each backend gets its own `AgentPredictor` instance. At routing time, the router
can:
- compute backend-specific features (notably `kvmatch_text` via the router prefix cache),
- run *all* backend predictors to obtain per-backend scores,
- then (for now) still pick one backend via the current policy (round-robin),
- and update only the chosen backend predictor with observed outcomes.

Cache ratio
----------
We deterministically set:

    pred_cache_ratio := clamp(kvmatch_text, 0, 1)

and provide it both:
- as a logged prediction output ("cache_ratio"), and
- as an input feature ("pred_cache_ratio") to latency/cost/perf models.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from river import compose, preprocessing, tree


Features = Dict[str, Any]
MetricPred = Tuple[float, float]
Predictions = Dict[str, MetricPred]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True, slots=True)
class PredictorInput:
    """Inputs known at routing time."""

    backend_id: str

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

    Cache ratio:
    - derived deterministically from kvmatch_text.
    """

    def __init__(self) -> None:
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
            # NOTE: kept for compatibility; in the new design, each backend
            # has its own predictor instance, so this feature is constant
            # per predictor and does not cause cross-backend interference.
            "backend_id": inp.backend_id,
            "model": inp.model,
            "source": inp.source,
            "turn_number": float(inp.turn_number),
            "prompt_chars": float(len(inp.prompt_repr)),
            "kvmatch_text": float(inp.kvmatch_text),
            "router_inflight": float(inp.router_inflight),
            "router_rps_1s": float(inp.router_rps_1s),
        }

    def _pred_cache_ratio(self, inp: PredictorInput) -> float:
        """
        Deterministic cache-ratio "prediction" from router-known text prefix match.
        """

        return _clamp01(inp.kvmatch_text)

    def predict(self, inp: PredictorInput) -> Predictions:
        x_base = self._base_features(inp)
        pred_cache_ratio = self._pred_cache_ratio(inp)
        x = {**x_base, "pred_cache_ratio": float(pred_cache_ratio)}

        lat = self.model_latency.predict_one(x)
        cost = self.model_cost.predict_one(x)

        lat_f = max(0.0, float(lat) if lat is not None else 0.0)
        cost_f = max(0.0, float(cost) if cost is not None else 0.0)

        proba = self.model_perf.predict_proba_one(x)
        perf_f = float(proba.get(True, 0.0)) if isinstance(proba, Mapping) else 0.0
        perf_f = _clamp01(perf_f)

        dummy_std = 0.0

        return {
            "latency_ms": (lat_f, dummy_std),
            "cost_tokens": (cost_f, dummy_std),
            "performance": (perf_f, dummy_std),
            # For logging and analysis
            "cache_ratio": (float(pred_cache_ratio), dummy_std),
        }

    def update(
        self,
        inp: PredictorInput,
        *,
        real_latency_ms: float,
        real_cost_tokens: int,
        real_perf_correct: bool,
    ) -> None:
        """
        Online update for latency/cost/perf models.

        Notes
        -----
        We include `pred_cache_ratio` as an input feature, but it is computed
        deterministically from kvmatch_text.
        """

        x_base = self._base_features(inp)
        pred_cache_ratio = self._pred_cache_ratio(inp)
        x = {**x_base, "pred_cache_ratio": float(pred_cache_ratio)}

        self.model_latency.learn_one(x, float(real_latency_ms))
        self.model_cost.learn_one(x, float(real_cost_tokens))
        self.model_perf.learn_one(x, bool(real_perf_correct))


class AsyncBackendPredictorPool:
    """
    A set of independent predictors, one per backend_id, with per-backend locks.

    This prevents training data from different backends from interfering, while
    still allowing the router to obtain scores for *all* backends for each request.
    """

    def __init__(self, backend_ids: Iterable[str]) -> None:
        ids = [str(b).strip() for b in backend_ids if str(b).strip()]
        if not ids:
            raise ValueError(
                "AsyncBackendPredictorPool requires at least one backend_id"
            )

        # Preserve order but ensure uniqueness.
        seen: set[str] = set()
        uniq: list[str] = []
        for b in ids:
            if b in seen:
                continue
            seen.add(b)
            uniq.append(b)

        self._backend_ids: Tuple[str, ...] = tuple(uniq)
        self._predictors: dict[str, AgentPredictor] = {
            b: AgentPredictor() for b in uniq
        }
        self._locks: dict[str, asyncio.Lock] = {b: asyncio.Lock() for b in uniq}

    @property
    def backend_ids(self) -> Tuple[str, ...]:
        """Backend IDs managed by this pool (stable order)."""

        return self._backend_ids

    def _get(self, backend_id: str) -> tuple[AgentPredictor, asyncio.Lock]:
        bid = str(backend_id).strip()
        if bid not in self._predictors:
            raise KeyError(f"Unknown backend_id for predictor pool: {bid!r}")

        return self._predictors[bid], self._locks[bid]

    async def predict_one(self, inp: PredictorInput) -> Predictions:
        """Predict metrics for exactly one backend (from inp.backend_id)."""

        pred, lock = self._get(inp.backend_id)
        async with lock:
            return pred.predict(inp)

    async def predict_all(
        self, inputs_by_backend: Mapping[str, PredictorInput]
    ) -> Dict[str, Predictions]:
        """
        Predict metrics for all provided backends concurrently.

        Parameters
        ----------
        inputs_by_backend
            Mapping from backend_id -> PredictorInput.

        Returns
        -------
        Dict[str, Predictions]
            Mapping from backend_id -> Predictions.
        """

        async def _predict_under_lock(
            backend_id: str, inp: PredictorInput
        ) -> tuple[str, Predictions]:
            pred, lock = self._get(backend_id)
            async with lock:
                return backend_id, pred.predict(inp)

        tasks = [
            _predict_under_lock(backend_id, inp)
            for backend_id, inp in inputs_by_backend.items()
        ]
        results = await asyncio.gather(*tasks)
        return {backend_id: preds for backend_id, preds in results}

    async def update_one(
        self,
        inp: PredictorInput,
        *,
        real_latency_ms: float,
        real_cost_tokens: int,
        real_perf_correct: bool,
    ) -> None:
        """Update exactly one backend predictor (from inp.backend_id)."""

        pred, lock = self._get(inp.backend_id)
        async with lock:
            pred.update(
                inp,
                real_latency_ms=float(real_latency_ms),
                real_cost_tokens=int(real_cost_tokens),
                real_perf_correct=bool(real_perf_correct),
            )
