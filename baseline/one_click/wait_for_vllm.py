#!/usr/bin/env python3
"""Wait for all configured vLLM endpoints to become healthy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


def is_healthy(url: str) -> bool:
    try:
        headers = {"User-Agent": "one-click"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return False
            _ = resp.read()
            return True
    except Exception:
        return False


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Can't signal it, but it exists.
        return True


def tail_text(path: Path, max_lines: int = 160) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<failed to read log {path}: {e}>"
    lines = txt.splitlines()
    if len(lines) <= max_lines:
        return txt
    return "\n".join(lines[-max_lines:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run directory created by one-click (contains vllm_pids.txt and vllm_*.log).",
    )
    ap.add_argument("--timeout-sec", type=int, default=600)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    vllm_cfg = cfg.get("vllm") or {}
    host = str(vllm_cfg.get("host") or "127.0.0.1")

    run_dir = Path(args.run_dir).resolve()
    models = list(vllm_cfg.get("models") or [])
    ports = [int(m["port"]) for m in models]
    urls = [f"http://{host}:{p}/v1/models" for p in ports]

    # Map PIDs to models by launch order (same order as config).
    pid_file = run_dir / "vllm_pids.txt"
    pids: list[int] = []
    if pid_file.exists():
        for line in pid_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pids.append(int(line))
            except ValueError:
                continue

    if pids and len(pids) != len(models):
        print(
            f"[wait_for_vllm] Warning: pid count ({len(pids)}) != model count ({len(models)}). ",
            file=sys.stderr,
        )

    deadline = time.time() + int(args.timeout_sec)

    pending = set(urls)
    while pending and time.time() < deadline:
        # Fail fast if any launched vLLM process has exited.
        for idx, m in enumerate(models):
            if idx >= len(pids):
                continue
            pid = pids[idx]
            if not is_pid_alive(pid):
                name = str(
                    m.get("name") or m.get("served_model_name") or f"model_{idx}"
                )
                log_path = run_dir / f"vllm_{name}.log"
                print(
                    f"[wait_for_vllm] vLLM process exited early: name={name} pid={pid}",
                    file=sys.stderr,
                )
                if log_path.exists():
                    print(f"[wait_for_vllm] ---- tail {log_path} ----", file=sys.stderr)
                    print(tail_text(log_path), file=sys.stderr)
                else:
                    print(
                        f"[wait_for_vllm] Log file not found: {log_path}",
                        file=sys.stderr,
                    )
                raise SystemExit(1)

        for u in list(pending):
            if is_healthy(u):
                pending.remove(u)
        if pending:
            time.sleep(1)

    if pending:
        raise SystemExit(
            f"vLLM health check timed out. Still pending: {sorted(pending)}"
        )

    print("[wait_for_vllm] All endpoints healthy")


if __name__ == "__main__":
    main()
