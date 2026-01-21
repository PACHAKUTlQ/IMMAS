"""
immas.analysis.analyzer_outliers

Outlier and sanity-check reporting:
- turn-number continuity warnings
- top latency cases
- residual outliers (latency/cost/perf)
- inconsistent usage cases

KV mismatch deep-dive is intentionally not surfaced by default anymore, since
pred_cache_ratio is now deterministic from router-side text prefix match.
"""

from __future__ import annotations

import math

from typing import Any, List, Mapping, Sequence, Tuple

from immas.analysis.utils import _f, _i, _s, _short_id, _is_finite


def _turn_number_gaps(
    by_did: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Tuple[str, List[int]]]:
    gaps: List[Tuple[str, List[int]]] = []
    for did, rs in by_did.items():
        turns = sorted(
            {_i(r.get("turn_number")) for r in rs if _i(r.get("turn_number")) > 0}
        )
        if not turns:
            continue
        expected = list(range(turns[0], turns[-1] + 1))
        if turns != expected:
            gaps.append((did, turns))
    return gaps


def _print_turn_number_gaps(gaps: Sequence[Tuple[str, List[int]]]) -> None:
    if gaps:
        print(
            "\nWARNING: turn-number gaps/duplicates detected (may indicate retries or partial runs)"
        )
        print(
            "--------------------------------------------------------------------------"
        )
        for did, turns in gaps[:10]:
            print(f"did={_short_id(did)} turns={turns}")
        if len(gaps) > 10:
            print(f"... {len(gaps) - 10} more")
    else:
        print("\nTurn-number continuity: OK (per dialogue_id)")


def _top_latency_outliers(
    ok_by_end: Sequence[Mapping[str, Any]], *, topk: int
) -> List[Mapping[str, Any]]:
    return sorted(ok_by_end, key=lambda r: _f(r.get("obs_latency_ms")), reverse=True)[
        :topk
    ]


def _print_top_latency_outliers(top_lat: Sequence[Mapping[str, Any]]) -> None:
    print("\nTop latency outliers (with context)")
    print("----------------------------------")
    for r in top_lat:
        backend_id = _s(r.get("backend_id"))
        model = _s(r.get("model"))
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)

        obs_ms = _f(r.get("obs_latency_ms"))
        pred_ms = _f(r.get("pred_latency_ms"))

        pr_cache = _f(r.get("pred_cache_ratio"))
        ocr = _f(r.get("obs_cache_ratio"))

        pred_cost = _f(r.get("pred_cost_tokens"))
        obs_cost = _f(r.get("obs_total_tokens"))

        pred_perf = _f(r.get("pred_perf_prob"))
        correct = bool(r.get("correct", True))

        inflight = _i(r.get("router_inflight"))
        rps = _f(r.get("router_rps_1s"))

        print(
            f"backend={backend_id} model={model} did={did} turn={turn} "
            f"obs_ms={obs_ms:.1f} pred_ms={pred_ms:.1f} "
            f"pred_cache={pr_cache:.3f} obs_cache={ocr:.3f} "
            f"pred_cost={pred_cost:.1f} obs_cost={obs_cost:.1f} "
            f"pred_perf={pred_perf:.3f} correct={int(correct)} "
            f"inflight={inflight} rps_1s={rps:.1f}"
        )


def _print_residual_outliers(
    ok_by_end: Sequence[Mapping[str, Any]], *, topk: int
) -> None:
    # Residual outliers (obs - pred)
    lat_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    cost_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    perf_residuals: List[Tuple[float, Mapping[str, Any]]] = []

    for r in ok_by_end:
        # Latency residuals
        ol = _f(r.get("obs_latency_ms"), math.nan)
        pl = _f(r.get("pred_latency_ms"), math.nan)
        if _is_finite(ol) and _is_finite(pl):
            lat_residuals.append((ol - pl, r))

        # Cost residuals (tokens)
        oc = _f(r.get("obs_total_tokens"), math.nan)
        pc = _f(r.get("pred_cost_tokens"), math.nan)
        if _is_finite(oc) and _is_finite(pc):
            cost_residuals.append((oc - pc, r))

        # Performance residuals: y - p where y in {0,1}
        pp = _f(r.get("pred_perf_prob"), math.nan)
        if _is_finite(pp):
            y = 1.0 if bool(r.get("correct", True)) else 0.0
            perf_residuals.append((y - pp, r))

    lat_residuals_sorted = sorted(lat_residuals, key=lambda x: abs(x[0]), reverse=True)[
        :topk
    ]
    cost_residuals_sorted = sorted(
        cost_residuals, key=lambda x: abs(x[0]), reverse=True
    )[:topk]
    perf_residuals_sorted = sorted(
        perf_residuals, key=lambda x: abs(x[0]), reverse=True
    )[:topk]

    def _ctx(r: Mapping[str, Any]) -> str:
        backend_id = _s(r.get("backend_id"))
        model = _s(r.get("model"))
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        return f"backend={backend_id} model={model} did={did} turn={turn}"

    if lat_residuals_sorted:
        print("\nTop |latency residual| outliers (obs - pred)")
        print("--------------------------------------------")
        for resid, r in lat_residuals_sorted:
            obs_ms = _f(r.get("obs_latency_ms"))
            pred_ms = _f(r.get("pred_latency_ms"))
            print(
                f"{_ctx(r)} resid_ms={resid:.1f} obs_ms={obs_ms:.1f} pred_ms={
                    pred_ms:.1f}"
            )

    if cost_residuals_sorted:
        print("\nTop |cost residual| outliers (obs - pred, tokens)")
        print("------------------------------------------------")
        for resid, r in cost_residuals_sorted:
            obs_tok = _f(r.get("obs_total_tokens"))
            pred_tok = _f(r.get("pred_cost_tokens"))
            print(
                f"{_ctx(r)} resid_tok={resid:+.1f} obs_tok={obs_tok:.1f} pred_tok={
                    pred_tok:.1f}"
            )

    if perf_residuals_sorted:
        print("\nTop |performance residual| outliers (y - p)")
        print("------------------------------------------")
        for resid, r in perf_residuals_sorted:
            pp = _f(r.get("pred_perf_prob"))
            y = 1 if bool(r.get("correct", True)) else 0
            print(f"{_ctx(r)} resid={resid:+.3f} y={y} pred_perf_prob={pp:.3f}")


def _find_inconsistent_usage_cases(
    ok_by_end: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    inconsistent: List[Mapping[str, Any]] = []
    for r in ok_by_end:
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        if pt > 0 and ct > pt:
            inconsistent.append(r)
    return inconsistent


def _print_inconsistent_usage_cases(
    inconsistent: Sequence[Mapping[str, Any]], *, topk: int
) -> None:
    if not inconsistent:
        return

    print("\nWARNING: inconsistent usage (obs_cached_tokens > obs_prompt_tokens)")
    print("---------------------------------------------------------------")
    for r in inconsistent[:topk]:
        backend_id = _s(r.get("backend_id"))
        model = _s(r.get("model"))
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        ocr = _f(r.get("obs_cache_ratio"))
        print(
            f"backend={backend_id} model={model} did={did} turn={turn} "
            f"prompt_tok={pt} cached_tok={ct} obs_cr={ocr:.3f}"
        )
