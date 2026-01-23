"""
immas.router.auction.mcmf

A small, dependency-free min-cost max-flow (MCMF) solver suitable for the router's
micro-batch sizes.

We implement the successive shortest augmenting path algorithm with Johnson-style
potentials:
- Bellman-Ford for initial potentials (handles negative edge costs),
- Dijkstra on reduced costs for each augmentation.

This is efficient enough for typical router batch sizes (e.g. 8-128) and a small
number of backends.

References
----------
- Successive shortest augmenting path algorithm for min-cost flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Tuple

_INF: int = 10**18


@dataclass(slots=True)
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int


def _add_edge(g: List[List[_Edge]], fr: int, to: int, cap: int, cost: int) -> None:
    """
    Add a directed edge (fr -> to) with capacity and cost, plus the reverse edge.
    """
    fwd = _Edge(to=to, rev=len(g[to]), cap=int(cap), cost=int(cost))
    rev = _Edge(to=fr, rev=len(g[fr]), cap=0, cost=-int(cost))
    g[fr].append(fwd)
    g[to].append(rev)


def min_cost_flow(
    n: int,
    *,
    edges: List[Tuple[int, int, int, int]],
    s: int,
    t: int,
    max_flow: int,
) -> Tuple[int, int, List[List[_Edge]]]:
    """
    Compute min-cost flow up to `max_flow`.

    Parameters
    ----------
    n
        Number of nodes.
    edges
        List of edges as (u, v, cap, cost).
    s
        Source node id.
    t
        Sink node id.
    max_flow
        Desired flow amount (upper bound). Algorithm may return less if infeasible.

    Returns
    -------
    (flow, cost, graph)
        flow: achieved flow
        cost: total min cost of the achieved flow
        graph: residual graph (can be inspected to extract the matching)
    """

    if n <= 0:
        return 0, 0, [[]]
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError("Invalid source/sink node ids")
    if max_flow <= 0:
        g: List[List[_Edge]] = [[] for _ in range(n)]
        for u, v, cap, cost in edges:
            _add_edge(g, u, v, cap, cost)
        return 0, 0, g

    g = [[] for _ in range(n)]
    for u, v, cap, cost in edges:
        if cap <= 0:
            continue
        _add_edge(g, int(u), int(v), int(cap), int(cost))

    # Initial potentials via Bellman-Ford (handles negative costs).
    h = [0] * n
    dist = [_INF] * n
    dist[s] = 0
    for _ in range(n - 1):
        updated = False
        for v in range(n):
            if dist[v] == _INF:
                continue
            dv = dist[v]
            for e in g[v]:
                if e.cap <= 0:
                    continue
                nd = dv + e.cost
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    updated = True
        if not updated:
            break
    for i in range(n):
        if dist[i] != _INF:
            h[i] = dist[i]

    flow = 0
    cost = 0

    prevv = [-1] * n
    preve = [-1] * n

    while flow < max_flow:
        # Dijkstra on reduced costs.
        dist2 = [_INF] * n
        dist2[s] = 0
        pq: list[tuple[int, int]] = [(0, s)]

        while pq:
            d, v = heapq.heappop(pq)
            if d != dist2[v]:
                continue
            for i, e in enumerate(g[v]):
                if e.cap <= 0:
                    continue
                nd = d + e.cost + h[v] - h[e.to]
                if nd < dist2[e.to]:
                    dist2[e.to] = nd
                    prevv[e.to] = v
                    preve[e.to] = i
                    heapq.heappush(pq, (nd, e.to))

        if dist2[t] == _INF:
            break

        for v in range(n):
            if dist2[v] < _INF:
                h[v] += dist2[v]

        # Find bottleneck.
        addf = max_flow - flow
        v = t
        while v != s:
            pv = prevv[v]
            pe = preve[v]
            if pv < 0 or pe < 0:
                addf = 0
                break
            addf = min(addf, g[pv][pe].cap)
            v = pv

        if addf <= 0:
            break

        # Augment.
        v = t
        while v != s:
            pv = prevv[v]
            pe = preve[v]
            e = g[pv][pe]
            e.cap -= addf
            g[v][e.rev].cap += addf
            v = pv

        flow += addf
        cost += addf * h[t]

    return int(flow), int(cost), g
