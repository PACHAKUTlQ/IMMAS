"""
immas.analysis.analyzer_outliers

Outlier and sanity-check reporting:
- turn-number continuity warnings
- top latency cases
- residual outliers
- suspicious KV mismatch cases
- inconsistent usage cases

Logic is copied from the original run_analyzer.py.
"""

from __future__ import annotations

import math

from typing import Any, List, Mapping, Sequence, Tuple

from immas.analysis.utils import _f, _i, _s, _short_id


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
    print("\nTop latency outliers (with cache context)")
    print("----------------------------------------")
    for r in top_lat:
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        obs_ms = _f(r.get("obs_latency_ms"))
        pred_ms = _f(r.get("pred_latency_ms"))
        kv = _f(r.get("kvmatch_text"))
        pr = _f(r.get("pred_cache_ratio"))
        ocr = _f(r.get("obs_cache_ratio"))
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        inflight = _i(r.get("router_inflight"))
        rps = _f(r.get("router_rps_1s"))
        print(
            f"did={did} turn={turn} obs_ms={obs_ms:.1f} pred_ms={pred_ms:.1f} "
            f"kvmatch={kv:.3f} pred_cr={pr:.3f} obs_cr={ocr:.3f} "
            f"prompt_tok={pt} cached_tok={ct} inflight={inflight} rps_1s={rps:.1f}"
        )


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _print_residual_outliers(
    ok_by_end: Sequence[Mapping[str, Any]], *, topk: int
) -> None:
    # Residual outliers (obs - pred)
    lat_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    cache_residuals: List[Tuple[float, Mapping[str, Any]]] = []
    for r in ok_by_end:
        ol = _f(r.get("obs_latency_ms"), math.nan)
        pl = _f(r.get("pred_latency_ms"), math.nan)
        if _is_finite(ol) and _is_finite(pl):
            lat_residuals.append((ol - pl, r))

        oc = _f(r.get("obs_cache_ratio"), math.nan)
        pc = _f(r.get("pred_cache_ratio"), math.nan)
        if _is_finite(oc) and _is_finite(pc):
            cache_residuals.append((oc - pc, r))

    lat_residuals_sorted = sorted(lat_residuals, key=lambda x: abs(x[0]), reverse=True)[
        :topk
    ]
    cache_residuals_sorted = sorted(
        cache_residuals, key=lambda x: abs(x[0]), reverse=True
    )[:topk]

    if lat_residuals_sorted:
        print("\nTop |latency residual| outliers (obs - pred, with cache context)")
        print("---------------------------------------------------------------")
        for resid, r in lat_residuals_sorted:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            obs_ms = _f(r.get("obs_latency_ms"))
            pred_ms = _f(r.get("pred_latency_ms"))
            kv = _f(r.get("kvmatch_text"))
            ocr = _f(r.get("obs_cache_ratio"))
            pt = _i(r.get("obs_prompt_tokens"))
            ct = _i(r.get("obs_cached_tokens"))
            print(
                f"did={did} turn={turn} resid_ms={resid:.1f} obs_ms={obs_ms:.1f} pred_ms={pred_ms:.1f} "
                f"obs_cr={ocr:.3f} kvmatch={kv:.3f} prompt_tok={pt} cached_tok={ct}"
            )

    if cache_residuals_sorted:
        print("\nTop |cache_ratio residual| outliers (obs - pred)")
        print("------------------------------------------------")
        for resid, r in cache_residuals_sorted:
            did = _short_id(_s(r.get("dialogue_id")))
            turn = _i(r.get("turn_number"), -1)
            oc = _f(r.get("obs_cache_ratio"))
            pc = _f(r.get("pred_cache_ratio"))
            kv = _f(r.get("kvmatch_text"))
            lcp = _i(r.get("kvmatch_lcp_chars"))
            print(
                f"did={did} turn={turn} resid_cr={resid:+.3f} obs_cr={oc:.3f} pred_cr={pc:.3f} "
                f"kvmatch={kv:.3f} lcp_chars={lcp}"
            )


def _find_suspicious_kv_cases(
    ok_by_end: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    suspicious: List[Mapping[str, Any]] = []
    for r in ok_by_end:
        kv = _f(r.get("kvmatch_text"), math.nan)
        ocr = _f(r.get("obs_cache_ratio"), math.nan)
        if not (_is_finite(kv) and _is_finite(ocr)):
            continue
        if (
            abs(kv - ocr) >= 0.7
            or (kv >= 0.9 and ocr <= 0.1)
            or (kv <= 0.1 and ocr >= 0.9)
        ):
            suspicious.append(r)
    return suspicious


def _print_suspicious_kv_cases(
    suspicious: Sequence[Mapping[str, Any]], *, topk: int
) -> None:
    if not suspicious:
        return

    print("\nSuspicious KV cases (kvmatch_text vs obs_cache_ratio mismatch)")
    print("------------------------------------------------------------")
    for r in suspicious[:topk]:
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        kv = _f(r.get("kvmatch_text"))
        ocr = _f(r.get("obs_cache_ratio"))
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        lcp = _i(r.get("kvmatch_lcp_chars"))
        cached_chars = _i(r.get("cached_prompt_chars"))
        prompt_chars = _i(r.get("prompt_chars"))
        print(
            f"did={did} turn={turn} kvmatch={kv:.3f} obs_cr={ocr:.3f} "
            f"prompt_tok={pt} cached_tok={ct} "
            f"lcp_chars={lcp} cached_chars={cached_chars} prompt_chars={prompt_chars}"
        )


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
        did = _short_id(_s(r.get("dialogue_id")))
        turn = _i(r.get("turn_number"), -1)
        pt = _i(r.get("obs_prompt_tokens"))
        ct = _i(r.get("obs_cached_tokens"))
        ocr = _f(r.get("obs_cache_ratio"))
        print(f"did={did} turn={turn} prompt_tok={pt} cached_tok={ct} obs_cr={ocr:.3f}")
