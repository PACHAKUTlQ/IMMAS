"""
immas.router.auction.mechanism

Auction + VCG payments for routing a micro-batch.

This module is designed to be consistent with the reference simulation code.

Key alignment points
--------------------
- Welfare is computed from (L, C, P) with the same structure:
  client_valuation = δ * (P * quality_scale) - (1-δ) * (L * latency_scale)
  welfare = client_valuation - C

- Allocation is solved via min-cost max-flow (MCMF) by minimizing -welfare.

- VCG payments (two-part tariff) for each matched task i:
  payment_i = base_cost_i + max(0, externality_i)
  externality_i = W(S_{-i}) - (W(S) - w_{i,s(i)})

Notes on "cost" units
---------------------
In the simulation, C is in $ units. In the router, we only have a predicted token-cost
proxy. We therefore define "scaled base cost" as:

    base_cost = cost_scale * pred_cost_tokens

This makes welfare and payments consistent in the same "utility units".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from immas.router.auction.mcmf import min_cost_flow


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True, slots=True)
class AuctionParams:
    """
    Welfare / solver parameters.

    Defaults are chosen to mirror the reference simulation structure.
    """

    quality_scale: float = 100.0
    latency_scale: float = 5.0

    # Maps predicted token cost to the welfare/payment unit.
    cost_scale: float = 0.02

    # Client preference δ in [0,1].
    delta_default: float = 0.5

    # Only edges with welfare > min_welfare_edge are considered "auction-feasible",
    # matching the simulation code's `if welfare > 0: add edge`.
    min_welfare_edge: float = 0.0

    # Float -> int scale for min-cost flow costs:
    # cost_ij = int(round(-welfare_ij * mcmf_scale)).
    mcmf_scale: int = 1000


@dataclass(frozen=True, slots=True)
class VCGPayment:
    """
    VCG payment components (in the same "utility units" as welfare).
    """

    base_cost: float
    vcg_fee: float
    total_payment: float


@dataclass(frozen=True, slots=True)
class AuctionResult:
    """
    Result of running auction allocation + VCG.

    Attributes
    ----------
    assignment
        assignment[i] = backend_index if matched by auction, else None.
    total_welfare
        Total welfare achieved by the auction allocation (sum over matched tasks).
    payments
        payments[i] = VCGPayment for matched tasks; None otherwise.
    """

    assignment: List[Optional[int]]
    total_welfare: float
    payments: List[Optional[VCGPayment]]


def compute_client_valuation(
    *,
    delta: float,
    pred_latency_ms: float,
    pred_perf_prob: float,
    params: AuctionParams,
) -> float:
    """
    Compute the client's valuation part (excluding cost).

    L is treated as seconds (as in the reference code).
    """

    d = _clamp01(delta)
    L_s = max(0.0, float(pred_latency_ms)) / 1000.0
    P = _clamp01(pred_perf_prob)

    val_quality = float(params.quality_scale) * P
    val_latency = float(params.latency_scale) * L_s

    return float(d * val_quality - (1.0 - d) * val_latency)


def compute_scaled_base_cost(
    *, pred_cost_tokens: float, params: AuctionParams
) -> float:
    """
    Convert predicted token-cost proxy into the same unit used by welfare/payments.
    """

    c = max(0.0, float(pred_cost_tokens))

    return float(float(params.cost_scale) * c)


def compute_welfare(
    *,
    delta: float,
    pred_latency_ms: float,
    pred_cost_tokens: float,
    pred_perf_prob: float,
    params: AuctionParams,
) -> Tuple[float, float, float]:
    """
    Compute (welfare, client_valuation, base_cost) for a (task, backend) pair.
    """

    val = compute_client_valuation(
        delta=float(delta),
        pred_latency_ms=float(pred_latency_ms),
        pred_perf_prob=float(pred_perf_prob),
        params=params,
    )
    base_cost = compute_scaled_base_cost(
        pred_cost_tokens=float(pred_cost_tokens),
        params=params,
    )
    welfare = float(val - base_cost)

    return welfare, val, base_cost


def solve_allocation_mcmf(
    *,
    welfare: Sequence[Sequence[float]],
    capacities: Sequence[int],
    params: AuctionParams,
) -> Tuple[List[Optional[int]], float]:
    """
    Solve allocation by maximizing total welfare subject to capacities, considering
    only edges with welfare > params.min_welfare_edge.

    This matches the reference code which only adds edges when welfare > 0.
    """

    n_tasks = len(welfare)
    if n_tasks == 0:
        return [], 0.0
    n_backends = len(capacities)
    if n_backends == 0:
        return [None] * n_tasks, 0.0

    # Validate matrix.
    for i in range(n_tasks):
        if len(welfare[i]) != n_backends:
            raise ValueError("welfare matrix must be rectangular (tasks x backends)")

    caps = [max(0, int(c)) for c in capacities]
    total_cap = sum(caps)
    if total_cap <= 0:
        return [None] * n_tasks, 0.0

    scale = int(params.mcmf_scale)
    if scale <= 0:
        raise ValueError("params.mcmf_scale must be > 0")

    # Node layout:
    # 0: source
    # 1..n_tasks: task nodes
    # 1+n_tasks .. n_tasks+n_backends: backend nodes
    # last: sink
    s = 0
    task0 = 1
    back0 = task0 + n_tasks
    t = back0 + n_backends
    n_nodes = t + 1

    edges: list[tuple[int, int, int, int]] = []

    for i in range(n_tasks):
        edges.append((s, task0 + i, 1, 0))
    for j in range(n_backends):
        if caps[j] > 0:
            edges.append((back0 + j, t, caps[j], 0))

    # Only positive-welfare edges (paper/simulation consistent).
    for i in range(n_tasks):
        row = welfare[i]
        for j in range(n_backends):
            w = float(row[j])
            if w > float(params.min_welfare_edge):
                cost = int(round(-w * scale))
                edges.append((task0 + i, back0 + j, 1, cost))

    desired_flow = min(n_tasks, total_cap)
    flow, total_cost, g = min_cost_flow(
        n_nodes, edges=edges, s=s, t=t, max_flow=desired_flow
    )

    assignment: list[Optional[int]] = [None] * n_tasks
    for i in range(n_tasks):
        v = task0 + i
        for e in g[v]:
            if back0 <= e.to < back0 + n_backends and e.cap == 0:
                assignment[i] = int(e.to - back0)
                break

    total_welfare = -float(total_cost) / float(scale)
    _ = flow

    return assignment, float(total_welfare)


def run_auction_with_vcg(
    *,
    welfare: Sequence[Sequence[float]],
    base_cost: Sequence[Sequence[float]],
    capacities: Sequence[int],
    params: AuctionParams,
) -> AuctionResult:
    """
    Run allocation + VCG payments.

    VCG is computed for matched tasks only, consistent with the reference code.
    """

    n_tasks = len(welfare)
    if n_tasks == 0:
        return AuctionResult(assignment=[], total_welfare=0.0, payments=[])

    n_backends = len(capacities)
    for i in range(n_tasks):
        if len(welfare[i]) != n_backends or len(base_cost[i]) != n_backends:
            raise ValueError(
                "welfare/base_cost matrices must be rectangular and aligned"
            )

    assignment, total_w = solve_allocation_mcmf(
        welfare=welfare,
        capacities=capacities,
        params=params,
    )

    payments: list[Optional[VCGPayment]] = [None] * n_tasks

    # Compute VCG for each matched task i.
    for i, j in enumerate(assignment):
        if j is None:
            continue

        w_is = float(welfare[i][j])
        c_is = float(base_cost[i][j])

        # Counterfactual welfare without task i.
        welfare_minus_i = [row for k, row in enumerate(welfare) if k != i]
        _, w_without_i = solve_allocation_mcmf(
            welfare=welfare_minus_i,
            capacities=capacities,
            params=params,
        )

        externality = float(w_without_i - (float(total_w) - w_is))
        vcg_fee = max(0.0, externality)
        total_payment = float(c_is + vcg_fee)
        payments[i] = VCGPayment(
            base_cost=float(c_is),
            vcg_fee=float(vcg_fee),
            total_payment=float(total_payment),
        )

    return AuctionResult(
        assignment=list(assignment),
        total_welfare=float(total_w),
        payments=list(payments),
    )
