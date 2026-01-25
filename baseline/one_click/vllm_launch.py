#!/usr/bin/env python3
"""Launch multiple vLLM OpenAI-compatible servers based on config.

This script is invoked by baseline/run_one_click_experiment.sh.
It will:
- pick GPUs automatically (greedy by remaining free memory)
- pre-filter by vLLM's own constraint: free_mem >= gpu_memory_utilization * total_mem
- spawn one vLLM process per model
- write PIDs to <run_dir>/vllm_pids.txt and per-model logs

Notes:
- We intentionally bind via integer GPU indices. vLLM (0.13.0) is not compatible with CUDA_VISIBLE_DEVICES=GPU-UUID.
- We set CUDA_DEVICE_ORDER=PCI_BUS_ID to keep CUDA indexing stable on heterogeneous GPU nodes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def run(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
) -> subprocess.Popen:
    stdout = None
    if stdout_path is not None:
        stdout = stdout_path.open("wb")
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)


def query_gpus() -> List[Dict[str, Any]]:
    """Return [{'index','name','free_mb','total_mb'}, ...] or empty list if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows: List[Dict[str, Any]] = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "free_mb": int(parts[2]),
                    "total_mb": int(parts[3]),
                }
            )
        return rows
    except Exception:
        return []


def gpu_threshold_mb(
    gpu_row: Dict[str, Any],
    min_free_mb: Optional[int],
    gpu_memory_utilization: Optional[float],
) -> int:
    threshold = 0
    if min_free_mb is not None:
        threshold = max(threshold, int(min_free_mb))
    if gpu_memory_utilization is not None:
        threshold = max(
            threshold, int(float(gpu_memory_utilization) * int(gpu_row["total_mb"]))
        )
    return threshold


def pick_gpu(
    gpus: List[Dict[str, Any]],
    reserved_mb: Dict[int, int],
    min_free_mb: Optional[int],
    gpu_memory_utilization: Optional[float],
) -> int:
    if not gpus:
        return -1

    best_idx = -1
    best_remaining = -1

    eligible: List[Dict[str, Any]] = []
    for row in gpus:
        idx = int(row["index"])
        remaining = int(row["free_mb"]) - int(reserved_mb.get(idx, 0))
        if remaining >= gpu_threshold_mb(row, min_free_mb, gpu_memory_utilization):
            eligible.append(row)

    if not eligible:
        eligible = gpus

    for row in eligible:
        idx = int(row["index"])
        remaining = int(row["free_mb"]) - int(reserved_mb.get(idx, 0))
        if remaining > best_remaining:
            best_remaining = remaining
            best_idx = idx

    return best_idx


def build_vllm_cmd(method: str, host: str, m: Dict[str, Any]) -> List[str]:
    model_path = str(m["model_path"])
    served = str(m.get("served_model_name") or m.get("name"))
    port = int(m["port"])

    common = [
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served,
    ]

    if m.get("max_model_len"):
        common += ["--max-model-len", str(int(m["max_model_len"]))]
    if m.get("tensor_parallel_size"):
        common += ["--tensor-parallel-size", str(int(m["tensor_parallel_size"]))]
    # Force attention backend to avoid incompatible kernels on some CUDA stacks.
    # Valid values depend on vLLM build; common options include:
    # FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION
    attn_backend = m.get("attention_backend")
    if attn_backend:
        common += ["--attention-backend", str(attn_backend)]

    # Eager mode disables CUDA graph capture (helps stability on some drivers).
    if bool(m.get("enforce_eager")):
        common += ["--enforce-eager"]
    if m.get("gpu_memory_utilization") is not None:
        common += ["--gpu-memory-utilization", str(float(m["gpu_memory_utilization"]))]
    if m.get("max_num_seqs") is not None:
        common += ["--max-num-seqs", str(int(m["max_num_seqs"]))]

    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")

    if method == "vllm_serve":
        return ["vllm", "serve", model_path, "--api-key", api_key] + common

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--api-key",
        api_key,
    ] + common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    vllm_cfg = cfg.get("vllm") or {}
    models = list(vllm_cfg.get("models") or [])
    host = str(vllm_cfg.get("host") or "127.0.0.1")

    method = str(vllm_cfg.get("launch_method") or "auto")
    if method == "auto":
        method = "vllm_serve" if shutil_which("vllm") else "python_api_server"

    pids_file = run_dir / "vllm_pids.txt"
    gpus = query_gpus()
    reserved_mb: Dict[int, int] = {}

    procs: List[subprocess.Popen] = []

    for m in models:
        name = str(m.get("name") or m.get("served_model_name") or "model")

        gpu_cfg = m.get("gpu")
        pinned_gpu: Optional[int] = None
        if gpu_cfg is not None:
            if isinstance(gpu_cfg, int):
                pinned_gpu = int(gpu_cfg)
            elif isinstance(gpu_cfg, str):
                s = gpu_cfg.strip().lower()
                if s not in {"", "auto", "none"}:
                    pinned_gpu = int(gpu_cfg.strip())

        min_free_mb_raw = m.get("min_free_mem_mb")
        min_free_mb = int(min_free_mb_raw) if min_free_mb_raw is not None else None
        util_raw = m.get("gpu_memory_utilization")
        util = float(util_raw) if util_raw is not None else None

        if pinned_gpu is not None:
            gpu_id = pinned_gpu
        else:
            gpu_id = pick_gpu(gpus, reserved_mb, min_free_mb, util)

        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Reserve: by default allocate one GPU per model within this run.
            reserve_override = m.get("reserve_mem_mb")
            if reserve_override is not None:
                reserve_mb = int(reserve_override)
            else:
                row = next((r for r in gpus if int(r["index"]) == gpu_id), None)
                reserve_mb = int(row["total_mb"]) if row is not None else 0
            reserved_mb[gpu_id] = max(reserved_mb.get(gpu_id, 0), reserve_mb)

        cmd = build_vllm_cmd(
            "vllm_serve" if method == "vllm_serve" else "python_api_server", host, m
        )

        log_path = run_dir / f"vllm_{name}.log"
        print(
            f"[vllm_launch] Starting {name} on GPU={env.get('CUDA_VISIBLE_DEVICES', 'auto')} -> {log_path}"
        )
        print("[vllm_launch] CMD:", " ".join(shlex.quote(c) for c in cmd))

        p = run(cmd, env=env, stdout_path=log_path)
        procs.append(p)
        with pids_file.open("a", encoding="utf-8") as f:
            f.write(str(p.pid) + "\n")
        time.sleep(0.5)

    print(f"[vllm_launch] Started {len(procs)} vLLM processes; PIDs in {pids_file}")


def shutil_which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


if __name__ == "__main__":
    main()
