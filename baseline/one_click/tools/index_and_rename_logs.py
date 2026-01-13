#!/usr/bin/env python3
"""Index and rename baseline/logs runs.

Creates:
- baseline/logs/_RENAMED_MAP.json: {old: new}
- baseline/logs/_RUN_INDEX.md: readable table of runs

Renaming rule (default):
  <ts>__<dataset>__<router>__<models>__n<maxreq>__shadow0|1

This is intentionally pragmatic: runs may be incomplete; we best-effort parse
run_config.yaml / summary.json / run.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _slug(text: str, max_len: int = 32) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        return "na"
    return s[:max_len]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _extract_ts(run_dir: Path) -> str:
    # Prefer run.log first line timestamp (YYYY-MM-DD HH:MM:SS,ms)
    run_log = run_dir / "run.log"
    if run_log.exists():
        try:
            first = run_log.open("r", encoding="utf-8").readline().strip()
            m = re.search(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", first)
            if m:
                return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
        except Exception:
            pass

    # Fallback: directory name if already timestamp-like
    if re.match(r"^\d{8}_\d{6}$", run_dir.name):
        return run_dir.name

    # Fallback: mtime
    try:
        st = run_dir.stat()
        import datetime as _dt

        dt = _dt.datetime.fromtimestamp(st.st_mtime)
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return "unknown"


def _dataset_label(cfg: Dict[str, Any]) -> str:
    rs = (cfg.get("request_stream") or {})
    datasets = (rs.get("datasets") or [])
    if not datasets:
        return "no-ds"

    # If exactly one dataset, try infer split from path.
    if len(datasets) == 1 and isinstance(datasets[0], dict):
        ds = datasets[0]
        name = str(ds.get("name") or "ds")
        path = str(ds.get("path") or "")
        p = path.lower()
        if "coqa_validation_multiturn" in p:
            return "coqa-val"
        if "coqa_train_multiturn" in p:
            return "coqa-train"
        if "quac_validation_multiturn" in p:
            return "quac-val"
        if "quac_train_multiturn" in p:
            return "quac-train"
        return _slug(name, 24)

    # Multiple datasets
    names: List[str] = []
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        names.append(_slug(str(ds.get("name") or "ds"), 16))
    if not names:
        return "mix"
    if len(names) <= 3:
        return "mix-" + "+".join(names)
    return "mix-many"


def _models_label(cfg: Dict[str, Any]) -> str:
    vllm = (cfg.get("vllm") or {})
    models = (vllm.get("models") or [])
    if models:
        names = []
        for m in models:
            if isinstance(m, dict) and m.get("name"):
                names.append(_slug(str(m.get("name")), 12))
        if names:
            return "+".join(names[:3])

    # Fallback: candidates file name
    cand = str(cfg.get("llm_candidates_path") or "")
    if cand:
        return _slug(Path(cand).stem, 24)
    return "models"


def _router_label(cfg: Dict[str, Any]) -> str:
    router = (cfg.get("router") or {})
    return _slug(str(router.get("name") or "router"), 24)


def _max_requests(cfg: Dict[str, Any]) -> Optional[int]:
    rs = (cfg.get("request_stream") or {})
    mr = rs.get("max_requests")
    try:
        return int(mr)
    except Exception:
        return None


def _shadow_flag(cfg: Dict[str, Any]) -> int:
    sc = (cfg.get("shadow_compare") or {})
    return 1 if bool(sc.get("enabled", False)) else 0


def _safe_target_name(name: str, max_len: int = 140) -> str:
    name = re.sub(r"[^a-zA-Z0-9_+\-]+", "-", name)
    name = name.strip("-")
    if len(name) <= max_len:
        return name
    return name[:max_len]


def _collision_resolve(target: Path) -> Path:
    if not target.exists():
        return target
    base = target.name
    parent = target.parent
    for i in range(1, 1000):
        cand = parent / f"{base}__r{i}"
        if not cand.exists():
            return cand
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
    return parent / f"{base}__{h}"


def build_new_name(run_dir: Path, cfg: Dict[str, Any]) -> str:
    ts = _extract_ts(run_dir)
    ds = _dataset_label(cfg)
    router = _router_label(cfg)
    models = _models_label(cfg)
    n = _max_requests(cfg)
    shadow = _shadow_flag(cfg)

    parts = [ts, ds, router, models]
    name = "__".join(parts)
    if n is not None:
        name += f"__n{n}"
    name += f"__shadow{shadow}"

    return _safe_target_name(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="baseline/logs")
    ap.add_argument("--apply", action="store_true", help="Actually rename directories")
    ap.add_argument("--only", default="", help="Regex to select runs by old name")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    if not logs_dir.exists():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    only_re = re.compile(args.only) if args.only else None

    runs = [p for p in logs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    runs = sorted(runs, key=lambda p: p.name)

    mapping: Dict[str, str] = {}
    index_rows: List[Dict[str, Any]] = []

    for run in runs:
        if only_re and not only_re.search(run.name):
            continue
        cfg = _read_yaml(run / "run_config.yaml") or {}
        summary = _read_json(run / "summary.json") or {}

        new_name = build_new_name(run, cfg)
        mapping[run.name] = new_name

        index_rows.append(
            {
                "old": run.name,
                "new": new_name,
                "dataset": _dataset_label(cfg),
                "router": _router_label(cfg),
                "models": _models_label(cfg),
                "max_requests": _max_requests(cfg),
                "shadow": _shadow_flag(cfg),
                "error_rate": summary.get("error_rate"),
                "total_model_calls": summary.get("total_model_calls"),
            }
        )

    # Write index + mapping first (so even dry-run is useful)
    (logs_dir / "_RENAMED_MAP.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# baseline/logs runs index",
        "",
        "| old | new | dataset | router | models | max_requests | shadow | error_rate | total_model_calls |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in index_rows:
        md_lines.append(
            "| {old} | {new} | {dataset} | {router} | {models} | {max_requests} | {shadow} | {error_rate} | {total_model_calls} |".format(
                **{k: ("" if v is None else v) for k, v in r.items()}
            )
        )
    (logs_dir / "_RUN_INDEX.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if not args.apply:
        print(f"Wrote mapping + index under {logs_dir} (dry-run; no renames)")
        return

    # Apply renames
    for run in runs:
        if only_re and not only_re.search(run.name):
            continue
        old = run
        new_name = mapping.get(run.name)
        if not new_name or new_name == run.name:
            continue
        target = logs_dir / new_name
        target = _collision_resolve(target)
        os.rename(old, target)

    print(f"Renamed {len(mapping)} runs. See {logs_dir / '_RENAMED_MAP.json'}")


if __name__ == "__main__":
    main()
