"""
immas.router.auction.mechanism

Batch-level auction allocation for the router.

Core idea (aligned with the reference simulation code)
------------------------------------------------------
For each request (task) i and backend (agent) j, we compute a welfare contribution:

    w_{ij} = δ_i * (Q_scale * P_{ij})
           - (1 - δ_i) * (L_scale * L_{ij})
           - (C_scale * C_{ij})
           + (O_scale * o_{ij})
           - congestion_penalty

Then we solve:

    maximize   Σ_{i,j} x_{ij} w_{ij}
    subject to Σ_j x_{ij} = 1                       for each task i (best-effort)
               Σ_i x_{ij} <= capacity_j             for each backend j
               x_{ij} ∈ {0,1}

We implement this as min-cost max-flow by using edge costs:

    cost_{ij} = -w_{ij} * SCALE

Important Router Constraints
----------------------------
- In a real router, requests must not be dropped. If the capacity-constrained
  solution cannot assign everyone, we fall back to a second pass (no welfare
  threshold) and finally to per-request greedy selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from immas.router.auction.mcmf import min_cost_flow


@dataclass(frozen=True, slots=True)
class AuctionParams:
    """
    Parameters controlling welfare computation and assignment behavior.
    """

    quality_scale: float = 100.0
    latency_scale: float = 5.0
    cost_scale: float = 0.02
    overlap_scale: float = 10.0

    # If welfare <= min_welfare_edge, that edge is omitted in the first pass.
    # To preserve router liveness, we will still ensure at least one edge per task.
    min_welfare_edge: float = 0.0

    # Convert float welfare to int costs via: int(round(-welfare * mcmf_scale))
    mcmf_scale: int = 1000


def compute_welfare(
    *,
    delta: float,
    pred_latency_ms: float,
    pred_cost_tokens: float,
    pred_perf_prob: float,
    overlap: float,
    params: AuctionParams,
) -> float:
    """
    Compute welfare contribution for assigning one task to one backend.

    Notes
    -----
    - pred_perf_prob is treated as a "quality/probability" proxy in [0,1].
    - latency uses seconds internally to mirror the simulation's L in seconds.
    - pred_cost_tokens is a router token-cost proxy; cost_scale makes it comparable.
    """

    d = float(delta)
    d = max(0.0, min(1.0, d))

    L_s = max(0.0, float(pred_latency_ms)) / 1000.0
    C = max(0.0, float(pred_cost_tokens))
    P = max(0.0, min(1.0, float(pred_perf_prob)))
    o = max(0.0, min(1.0, float(overlap)))

    val_quality = params.quality_scale * P
    val_latency = params.latency_scale * L_s

    client_valuation = d * val_quality - (1.0 - d) * val_latency
    welfare = client_valuation - (params.cost_scale * C) + (params.overlap_scale * o)

    return float(welfare)


def solve_batch_assignment(
    *,
    welfare: Sequence[Sequence[float]],
    capacities: Sequence[int],
    params: AuctionParams,
) -> Tuple[List[Optional[int]], float]:
    """
    Solve the batch allocation problem.

    Parameters
    ----------
    welfare
        Matrix w[i][j] of welfare contributions (tasks x backends).
    capacities
        Backend capacities for this batch (length = #backends).
    params
        Auction parameters.

    Returns
    -------
    (assignment, total_welfare)
        assignment[i] = backend_index or None (best-effort; router should fallback)
        total_welfare = sum assigned welfare over the MCMF solution
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

    def _run_mcmf(*, use_threshold: bool) -> Tuple[List[Optional[int]], float, int]:
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

        # Source -> task (cap=1)
        for i in range(n_tasks):
            edges.append((s, task0 + i, 1, 0))

        # Backend -> sink (cap=capacity)
        for j in range(n_backends):
            if caps[j] > 0:
                edges.append((back0 + j, t, caps[j], 0))

        scale = int(params.mcmf_scale)
        if scale <= 0:
            raise ValueError("params.mcmf_scale must be > 0")

        # Task -> backend edges.
        for i in range(n_tasks):
            row = welfare[i]
            # Candidate set (first pass: welfare > threshold).
            if use_threshold:
                cand = [
                    j for j in range(n_backends) if row[j] > params.min_welfare_edge
                ]
                if not cand:
                    # Ensure at least one edge per task for liveness.
                    j_best = max(range(n_backends), key=lambda j: row[j])
                    cand = [j_best]
            else:
                cand = list(range(n_backends))

            for j in cand:
                w = float(row[j])
                cost = int(round(-w * scale))
                edges.append((task0 + i, back0 + j, 1, cost))

        desired_flow = min(n_tasks, total_cap)
        flow, total_cost, g = min_cost_flow(
            n_nodes, edges=edges, s=s, t=t, max_flow=desired_flow
        )

        assignment: list[Optional[int]] = [None] * n_tasks

        # Extract matching: look at residual edges out of task nodes.
        # We added cap=1 edges from task->backend. If that edge is saturated (cap==0),
        # then the reverse edge has cap==1.
        for i in range(n_tasks):
            v = task0 + i
            for e in g[v]:
                if not (back0 <= e.to < back0 + n_backends):
                    continue
                # If forward cap is 0, it was used.
                if e.cap == 0:
                    assignment[i] = int(e.to - back0)
                    break

        total_welfare = -float(total_cost) / float(scale)

        return assignment, total_welfare, int(flow)

    # Apply welfare threshold (paper-style positive-welfare edges).
    assignment, total_welfare, flow = _run_mcmf(use_threshold=True)

    # If we couldn't assign everyone, rerun without edge threshold.
    if flow < min(n_tasks, total_cap):
        assignment2, total_welfare2, flow2 = _run_mcmf(use_threshold=False)
        assignment, total_welfare, flow = assignment2, total_welfare2, flow2

    return assignment, float(total_welfare)
