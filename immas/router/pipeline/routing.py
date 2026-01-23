"""
immas.router.pipeline.routing

Routing logic for the router app.
"""

from __future__ import annotations

import asyncio

from typing import Optional

from immas.common.load import LoadSnapshot
from immas.router.auction.mechanism import AuctionParams, solve_batch_assignment
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

    if n <= 0:
        return []
    if not backends:
        return []

    async with rr_lock:
        start_idx: int = int(state.rr_index)
        chosen = [backends[(start_idx + i) % len(backends)] for i in range(n)]
        state.rr_index = (start_idx + n) % len(backends)

    return chosen


def _delta_for_request(*, state: RouterState) -> float:
    """
    Determine δ (quality-vs-latency preference) for a request.

    Currently we use a single default δ from config.
    This is the minimal, production-safe choice (paper structure is preserved).
    """

    d = float(state.cfg.router.auction.delta_default)
    return max(0.0, min(1.0, d))


async def select_backends_auction(
    state: RouterState,
    *,
    active_indices: list[int],
    backend_scores_by_req: list[list[RouterBackendScore]],
) -> list[Optional[HttpOpenAIBackend]]:
    """
    Auction routing: solve a batch assignment maximizing total welfare subject to
    backend capacities.

    Parameters
    ----------
    active_indices
        Indices of requests in the batch that are still pending (not failed early).
    backend_scores_by_req
        Per-request list of RouterBackendScore, ordered in the same order as
        state.backends.

    Returns
    -------
    list[Optional[HttpOpenAIBackend]]
        List aligned with the original batch length; None for inactive indices.
    """

    n_total = len(backend_scores_by_req)
    out: list[Optional[HttpOpenAIBackend]] = [None] * n_total
    if not active_indices:
        return out

    backends = state.backends
    if not backends:
        return out

    backend_ids = [b.backend_id for b in backends]
    n_backends = len(backends)

    # Snapshot per-backend inflight to derive effective capacities and congestion penalties.
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
        int(state.backend_capacity_by_id.get(bid, 1)) for bid in backend_ids
    ]
    cap_cfg_by_j = [max(1, c) for c in cap_cfg_by_j]

    # Available capacity now; if too small for the batch, fall back to configured capacity.
    avail_by_j = [max(0, cap_cfg_by_j[j] - inflight_by_j[j]) for j in range(n_backends)]
    if sum(avail_by_j) >= len(active_indices):
        capacities = avail_by_j
    else:
        capacities = cap_cfg_by_j

    auc_cfg = state.cfg.router.auction
    params = AuctionParams(
        quality_scale=float(auc_cfg.quality_scale),
        latency_scale=float(auc_cfg.latency_scale),
        cost_scale=float(auc_cfg.cost_scale),
        overlap_scale=float(auc_cfg.overlap_scale),
        min_welfare_edge=float(auc_cfg.min_welfare_edge),
        mcmf_scale=int(auc_cfg.mcmf_scale),
    )
    congestion_penalty = float(auc_cfg.congestion_penalty)

    # Build welfare matrix for active tasks only: shape (N_active x N_backends).
    delta = _delta_for_request(state=state)
    welfare: list[list[float]] = []

    for i in active_indices:
        scores = backend_scores_by_req[i]
        if len(scores) != n_backends:
            # Highly defensive; treat missing as zeros.
            row = [0.0] * n_backends
            welfare.append(row)
            continue

        row_w: list[float] = []
        for j, s in enumerate(scores):
            w = _compute_welfare_from_score(delta=delta, score=s, params=params)

            # Optional congestion term.
            if congestion_penalty > 0.0:
                denom = float(max(1, cap_cfg_by_j[j]))
                w -= congestion_penalty * (float(inflight_by_j[j]) / denom)

            row_w.append(float(w))
        welfare.append(row_w)

    assignment_active, _total_w = solve_batch_assignment(
        welfare=welfare,
        capacities=capacities,
        params=params,
    )

    # Map assignment back to original indices; best-effort fallback if None.
    for k, i in enumerate(active_indices):
        j = assignment_active[k]
        if j is None or not (0 <= j < n_backends):
            # Fallback: choose per-request best welfare backend (ignoring capacity).
            row = welfare[k]
            j = max(range(n_backends), key=lambda jj: row[jj])
        out[i] = backends[int(j)]

    return out


def _compute_welfare_from_score(
    *, delta: float, score: RouterBackendScore, params: AuctionParams
) -> float:
    """
    Convert RouterBackendScore fields into the auction's welfare computation.
    """

    from immas.router.auction.mechanism import compute_welfare

    return compute_welfare(
        delta=float(delta),
        pred_latency_ms=float(score.pred_latency_ms),
        pred_cost_tokens=float(score.pred_cost_tokens),
        pred_perf_prob=float(score.pred_perf_prob),
        overlap=float(score.kvmatch_text),
        params=params,
    )
