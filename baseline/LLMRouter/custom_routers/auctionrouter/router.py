"""
Auction Router - IMMAS-style welfare routing (baseline plugin).

This router uses IMMAS auction welfare computation with simple, configurable
heuristics for predicted latency/performance/cost.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class AuctionRouter(MetaRouter):
    """
    AuctionRouter - routes by maximizing auction welfare.

    Required config fields:
      data_path.llm_data: path to llm_candidates.json

    Optional hparam fields (with defaults):
      delta_default, quality_scale, latency_scale, cost_scale, min_welfare_edge, mcmf_scale
      size_default_b, perf_base, perf_per_b, perf_min, perf_max
      latency_base_ms, latency_per_b_ms, cost_tokens_per_b
      capacity_default, capacity_by_llm
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Ensure IMMAS repo root is importable when running from baseline/one_click.
        repo_root = Path(__file__).resolve().parents[4]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        try:
            from immas.router.auction.mechanism import (
                AuctionParams,
                compute_welfare,
                run_auction_with_vcg,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import IMMAS auction mechanism. Ensure IMMAS is importable in PYTHONPATH."
            ) from exc

        self._AuctionParams = AuctionParams
        self._compute_welfare = compute_welfare
        self._run_auction_with_vcg = run_auction_with_vcg

        if not getattr(self, "llm_data", None):
            raise ValueError(
                "No LLM data found. Please specify 'llm_data' in YAML config."
            )

        self.llm_names = list(self.llm_data.keys())
        if not self.llm_names:
            raise ValueError("LLM data is empty. At least one LLM is required.")

        hparam = self.cfg.get("hparam", {}) or {}

        self.params = self._AuctionParams(
            quality_scale=float(hparam.get("quality_scale", 100.0)),
            latency_scale=float(hparam.get("latency_scale", 5.0)),
            cost_scale=float(hparam.get("cost_scale", 0.02)),
            delta_default=float(hparam.get("delta_default", 0.5)),
            min_welfare_edge=float(hparam.get("min_welfare_edge", 0.0)),
            mcmf_scale=int(hparam.get("mcmf_scale", 1000)),
        )

        self.size_default = float(hparam.get("size_default_b", 7.0))
        self.perf_base = float(hparam.get("perf_base", 0.55))
        self.perf_per_b = float(hparam.get("perf_per_b", 0.005))
        self.perf_min = float(hparam.get("perf_min", 0.45))
        self.perf_max = float(hparam.get("perf_max", 0.95))

        self.latency_base_ms = float(hparam.get("latency_base_ms", 400.0))
        self.latency_per_b_ms = float(hparam.get("latency_per_b_ms", 12.0))
        self.cost_tokens_per_b = float(hparam.get("cost_tokens_per_b", 80.0))

        self.capacity_default = int(hparam.get("capacity_default", 1))
        self.capacity_by_llm = {
            str(k): int(v) for k, v in (hparam.get("capacity_by_llm") or {}).items()
        }

        print("✅ AuctionRouter initialized successfully")
        print(f"   Available LLMs: {', '.join(self.llm_names)}")

    @staticmethod
    def _parse_size_to_b(size_val: Any, default_b: float) -> float:
        if size_val is None:
            return float(default_b)
        if isinstance(size_val, (int, float)):
            return float(size_val)
        text = str(size_val)
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Bb]", text)
        if match:
            return float(match.group(1))
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
        return float(default_b)

    def _estimate_perf_prob(self, size_b: float) -> float:
        val = self.perf_base + self.perf_per_b * float(size_b)
        return float(max(self.perf_min, min(self.perf_max, val)))

    def _estimate_latency_ms(self, size_b: float) -> float:
        return float(self.latency_base_ms + self.latency_per_b_ms * float(size_b))

    def _estimate_cost_tokens(self, size_b: float) -> float:
        return float(self.cost_tokens_per_b * float(size_b))

    def _estimate_for_model(self, model_name: str) -> Dict[str, float]:
        meta = self.llm_data.get(model_name) or {}

        size_b = self._parse_size_to_b(
            meta.get("size") or meta.get("size_b"), self.size_default
        )

        perf_prob = meta.get("perf_prob")
        if perf_prob is None:
            perf_prob = meta.get("pred_perf_prob")
        if perf_prob is None:
            perf_prob = self._estimate_perf_prob(size_b)

        latency_ms = meta.get("latency_ms")
        if latency_ms is None:
            latency_ms = self._estimate_latency_ms(size_b)

        cost_tokens = meta.get("cost_tokens")
        if cost_tokens is None:
            cost_tokens = self._estimate_cost_tokens(size_b)

        return {
            "size_b": float(size_b),
            "perf_prob": float(perf_prob),
            "latency_ms": float(latency_ms),
            "cost_tokens": float(cost_tokens),
        }

    def _capacity(self, model_name: str) -> int:
        return int(self.capacity_by_llm.get(model_name, self.capacity_default))

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        delta = float(query_input.get("delta", self.params.delta_default))

        best_model = None
        best_welfare = -math.inf
        best_val = 0.0
        best_cost = 0.0
        model_scores: Dict[str, Dict[str, float]] = {}

        for name in self.llm_names:
            est = self._estimate_for_model(name)
            welfare, client_val, base_cost = self._compute_welfare(
                delta=delta,
                pred_latency_ms=est["latency_ms"],
                pred_cost_tokens=est["cost_tokens"],
                pred_perf_prob=est["perf_prob"],
                params=self.params,
            )
            model_scores[name] = {
                "welfare": float(welfare),
                "client_val": float(client_val),
                "base_cost": float(base_cost),
            }
            if welfare > best_welfare:
                best_model = name
                best_welfare = float(welfare)
                best_val = float(client_val)
                best_cost = float(base_cost)

        if best_model is None:
            best_model = self.llm_names[0]

        return {
            "query": query_input.get("query", ""),
            "model_name": best_model,
            "predicted_llm": best_model,
            "predicted_llm_name": best_model,
            "method": "auction",
            "welfare": float(best_welfare),
            "client_valuation": float(best_val),
            "base_cost": float(best_cost),
            "model_scores": model_scores,
        }

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch:
            return []

        capacities = [self._capacity(name) for name in self.llm_names]
        welfare_matrix: List[List[float]] = []
        base_cost_matrix: List[List[float]] = []
        score_cache: List[Dict[str, Dict[str, float]]] = []

        for item in batch:
            delta = float(item.get("delta", self.params.delta_default))
            row_welfare: List[float] = []
            row_base_cost: List[float] = []
            row_scores: Dict[str, Dict[str, float]] = {}

            for name in self.llm_names:
                est = self._estimate_for_model(name)
                welfare, client_val, base_cost = self._compute_welfare(
                    delta=delta,
                    pred_latency_ms=est["latency_ms"],
                    pred_cost_tokens=est["cost_tokens"],
                    pred_perf_prob=est["perf_prob"],
                    params=self.params,
                )
                row_welfare.append(float(welfare))
                row_base_cost.append(float(base_cost))
                row_scores[name] = {
                    "welfare": float(welfare),
                    "client_val": float(client_val),
                    "base_cost": float(base_cost),
                }

            welfare_matrix.append(row_welfare)
            base_cost_matrix.append(row_base_cost)
            score_cache.append(row_scores)

        result = self._run_auction_with_vcg(
            welfare=welfare_matrix,
            base_cost=base_cost_matrix,
            capacities=capacities,
            params=self.params,
        )

        outputs: List[Dict[str, Any]] = []
        for i, item in enumerate(batch):
            assigned_idx = result.assignment[i]
            if assigned_idx is None:
                # Fallback: max welfare in row
                row = welfare_matrix[i]
                best_j = int(max(range(len(row)), key=lambda j: row[j]))
                assigned_idx = best_j

            chosen = self.llm_names[int(assigned_idx)]
            scores = score_cache[i]
            chosen_score = scores.get(chosen, {})

            outputs.append(
                {
                    "query": item.get("query", ""),
                    "model_name": chosen,
                    "predicted_llm": chosen,
                    "predicted_llm_name": chosen,
                    "method": "auction",
                    "welfare": float(chosen_score.get("welfare", 0.0)),
                    "client_valuation": float(chosen_score.get("client_val", 0.0)),
                    "base_cost": float(chosen_score.get("base_cost", 0.0)),
                    "auction_total_welfare": float(result.total_welfare),
                    "model_scores": scores,
                }
            )

        return outputs

    def forward(self, batch):
        if isinstance(batch, list):
            return self.route_batch(batch)
        return self.route_single(batch)
