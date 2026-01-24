"""
immas.router.pipeline.routing

Routing logic for the router app.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Optional

from immas.common.load import LoadSnapshot
from immas.router.auction.mechanism import AuctionParams, run_auction_with_vcg
from immas.router.components.backend import HttpOpenAIBackend
from immas.router.components.logger import RouterBackendScore
from immas.router.state import RouterState


async def select_backends_round_robin(
    state: RouterState, n: int
) -> list[HttpOpenAIBackend]:
    """
    Select N backends in a single round-robin critical section.

    This is cheaper than acquiring the RR lock per request and also ensures
    deterministic batch ordering.
    """

    backends: list[HttpOpenAIBackend] = state.backends
    rr_lock: asyncio.Lock = state.rr_lock

    if n <= 0 or not backends:
        return []

    async with rr_lock:
        start_idx: int = int(state.rr_index)
        chosen = [backends[(start_idx + i) % len(backends)] for i in range(n)]
        state.rr_index = (start_idx + n) % len(backends)

    return chosen


@dataclass(frozen=True, slots=True)
class AuctionDecision:
    """
    Determine δ (quality-vs-latency preference) for a batch of requests.

    Contains per-request backend assignments, match flags, total welfare,
    and VCG-based fees and payments computed by the auction mechanism.
    """

    assigned_by_i: list[Optional[HttpOpenAIBackend]]
    auction_matched_by_i: list[bool]
    auction_total_welfare: float
    vcg_fee_by_i: list[Optional[float]]
    vcg_total_payment_by_i: list[Optional[float]]


async def select_backends_auction(
    state: RouterState,
    *,
    active_indices: list[int],
    backend_scores_by_req: list[list[RouterBackendScore]],
) -> AuctionDecision:
    """
    Run auction allocation + VCG for active requests.

    Unmatched tasks (no positive-welfare edges) are routed via fallback, but have:
    - auction_matched=False
    - no VCG payment
    """

    n_total = len(backend_scores_by_req)
    assigned: list[Optional[HttpOpenAIBackend]] = [None] * n_total
    matched: list[bool] = [False] * n_total
    vcg_fee_by_i: list[Optional[float]] = [None] * n_total
    vcg_pay_by_i: list[Optional[float]] = [None] * n_total

    if not active_indices or not state.backends:
        return AuctionDecision(
            assigned_by_i=assigned,
            auction_matched_by_i=matched,
            auction_total_welfare=0.0,
            vcg_fee_by_i=vcg_fee_by_i,
            vcg_total_payment_by_i=vcg_pay_by_i,
        )

    backends = state.backends

    backend_ids = [b.backend_id for b in backends]
    n_backends = len(backends)

    # Effective capacities: if we have enough "available", use that; otherwise fall back to configured cap.
    async def _snap(bid: str) -> LoadSnapshot:
        tr = state.backend_load_trackers.get(bid)
        if tr is None:
            # Defensive fallback
            return LoadSnapshot(
                inflight_requests=0, rps=0.0, window_s=1.0, t_monotonic=0.0
            )
        return await tr.snapshot()

    snaps = await asyncio.gather(
        *[_snap(bid) for bid in backend_ids], return_exceptions=False
    )
    inflight_by_j = [int(s.inflight_requests) for s in snaps]

    cap_cfg_by_j = [
        max(1, int(state.backend_capacity_by_id.get(bid, 1))) for bid in backend_ids
    ]
    avail_by_j = [max(0, cap_cfg_by_j[j] - inflight_by_j[j]) for j in range(n_backends)]
    capacities = avail_by_j if sum(avail_by_j) >= len(active_indices) else cap_cfg_by_j

    auc_cfg = state.cfg.router.auction
    params = AuctionParams(
        quality_scale=float(auc_cfg.quality_scale),
        latency_scale=float(auc_cfg.latency_scale),
        cost_scale=float(auc_cfg.cost_scale),
        delta_default=float(auc_cfg.delta_default),
        min_welfare_edge=float(auc_cfg.min_welfare_edge),
        mcmf_scale=int(auc_cfg.mcmf_scale),
    )

    # Build welfare/base_cost matrices (active tasks only) from RouterBackendScore
    welfare: list[list[float]] = []
    base_cost: list[list[float]] = []
    for i in active_indices:
        scores = backend_scores_by_req[i]
        if len(scores) != n_backends:
            welfare.append([0.0] * n_backends)
            base_cost.append([0.0] * n_backends)
            continue
        welfare.append([float(s.welfare) for s in scores])
        base_cost.append([float(s.base_cost) for s in scores])

    # Apply congestion penalty.
    congestion_penalty = float(auc_cfg.congestion_penalty)
    if congestion_penalty > 0.0:
        for i in range(len(welfare)):
            for j in range(n_backends):
                welfare[i][j] -= congestion_penalty * inflight_by_j[j] / cap_cfg_by_j[j]

    result = run_auction_with_vcg(
        welfare=welfare,
        base_cost=base_cost,
        capacities=capacities,
        params=params,
    )

    # Map assignment back to original indices; best-effort fallback if None.
    for k, i in enumerate(active_indices):
        j = result.assignment[k]
        if j is not None and 0 <= j < n_backends:
            assigned[i] = backends[int(j)]
            matched[i] = True
            pay = result.payments[k]
            if pay is not None:
                vcg_fee_by_i[i] = float(pay.vcg_fee)
                vcg_pay_by_i[i] = float(pay.total_payment)

    # Fallback for unmatched: choose per-request maximum welfare backend (even if <= 0).
    for i in active_indices:
        if assigned[i] is not None:
            continue
        scores = backend_scores_by_req[i]
        if not scores:
            continue
        j_best = max(range(len(scores)), key=lambda j: float(scores[j].welfare))
        assigned[i] = backends[int(j_best)]
        matched[i] = False

    return AuctionDecision(
        assigned_by_i=assigned,
        auction_matched_by_i=matched,
        auction_total_welfare=float(result.total_welfare),
        vcg_fee_by_i=vcg_fee_by_i,
        vcg_total_payment_by_i=vcg_pay_by_i,
    )
