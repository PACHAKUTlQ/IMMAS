#!/usr/bin/env bash
set -euo pipefail

# One-click experiment runner
# - Starts multiple vLLM OpenAI-compatible servers (optional)
# - Runs online many-requests-many-LLMs routing experiment
# - Saves logs/artifacts under baseline/logs/<run_id>/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$ROOT_DIR"
ONE_CLICK_DIR="$BASELINE_DIR/one_click"

CONFIG_PATH="${1:-$ONE_CLICK_DIR/configs/example_poisson.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

# Enable CUDA-safe multiprocessing for vLLM workers
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

# Make CUDA device ordering deterministic on heterogeneous GPU nodes.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

# Optional: use HF mirror if you need it
export HF_ENDPOINT=https://hf-mirror.com
# Some libraries look for this name instead.
export HF_HUB_ENDPOINT=${HF_HUB_ENDPOINT:-$HF_ENDPOINT}

# LiteLLM needs a non-empty key even for localhost; LLMRouter falls back to 'local'
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-local}

# LLMRouter expects API_KEYS (used for round-robin and provider selection).
# For local vLLM (OpenAI-compatible) runs, a single key is sufficient.
export API_KEYS=${API_KEYS:-$OPENAI_API_KEY}

PY_CMD=()

# Prefer running inside a specific conda env without requiring activation.
# Usage:
#   ONE_CLICK_CONDA_PREFIX=/path/to/env bash baseline/run_one_click_experiment.sh ...
ONE_CLICK_CONDA_PREFIX=${ONE_CLICK_CONDA_PREFIX:-${CONDA_PREFIX:-}}
if [[ -n "$ONE_CLICK_CONDA_PREFIX" ]] && [[ -x "${ONE_CLICK_CONDA_PREFIX}/bin/python" ]]; then
  PY_CMD=(/opt/miniconda/bin/conda run -p "$ONE_CLICK_CONDA_PREFIX" --no-capture-output python)
else
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY_CMD=($PYTHON_BIN)
  else
    if command -v python >/dev/null 2>&1; then
      PY_CMD=(python)
    else
      PY_CMD=(python3)
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Parse minimal fields from YAML (without requiring yq)
# -----------------------------------------------------------------------------
get_yaml_value() {
  local key="$1"
  # naive YAML accessor for simple 'key: value' (no nesting)
  # used only for a couple of top-level toggles
  grep -E "^${key}:" "$CONFIG_PATH" | head -n 1 | sed -E "s/^${key}:\s*//" | tr -d '"' | tr -d "'" || true
}

SKIP_VLLM_RAW="$(get_yaml_value "skip_vllm")"
SKIP_VLLM=${SKIP_VLLM:-${SKIP_VLLM_RAW:-false}}

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
RUN_DIR="$BASELINE_DIR/logs/$RUN_ID"
mkdir -p "$RUN_DIR"

cp "$CONFIG_PATH" "$RUN_DIR/run_config.yaml"

VLLM_PIDS_FILE="$RUN_DIR/vllm_pids.txt"
: > "$VLLM_PIDS_FILE"

cleanup() {
  if [[ -f "$VLLM_PIDS_FILE" ]]; then
    while IFS= read -r pid; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done < "$VLLM_PIDS_FILE"
  fi
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Auto-prepare multiturn datasets (CoQA/QuAC) if missing
# -----------------------------------------------------------------------------
# If your config references e.g. baseline/one_click/data/coqa_train_multiturn.jsonl
# and the file doesn't exist yet, we will download the HF dataset and build it.
# You can disable this with: SKIP_DATA_PREP=true
# You can force rebuild with: FORCE_DATA_PREP=true
# You can only run data prep then exit with: DATA_PREP_ONLY=true
SKIP_DATA_PREP=${SKIP_DATA_PREP:-false}
if [[ "$SKIP_DATA_PREP" != "true" ]]; then
  echo "[one-click] Preparing datasets (if needed)..."
  PROJECT_ROOT="$(cd "$BASELINE_DIR/.." && pwd)" \
  ONE_CLICK_PROJECT_ROOT="$PROJECT_ROOT" \
  ONE_CLICK_BASELINE_DIR="$BASELINE_DIR" \
  ONE_CLICK_CONFIG_PATH="$CONFIG_PATH" \
  "${PY_CMD[@]}" - <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

project_root = Path(os.environ["ONE_CLICK_PROJECT_ROOT"]).resolve()
baseline_dir = Path(os.environ["ONE_CLICK_BASELINE_DIR"]).resolve()
config_path = Path(os.environ["ONE_CLICK_CONFIG_PATH"]).resolve()

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
datasets = (((cfg.get("request_stream") or {}).get("datasets")) or [])

build_script = (baseline_dir / "one_click" / "data" / "build_multiturn_jsonl.py").resolve()
if not build_script.exists():
  print(f"[one-click][data] build script not found: {build_script}", file=sys.stderr)
  sys.exit(1)

requested_dataset_names = set()
for ds in datasets:
  if not isinstance(ds, dict):
    continue
  if ds.get("type") != "jsonl":
    continue
  name = str(ds.get("name") or "").lower()
  path = str(ds.get("path") or "").lower()
  if "coqa" in name or "coqa_" in path:
    requested_dataset_names.add("coqa")
  if "quac" in name or "quac_" in path:
    requested_dataset_names.add("quac")

splits = [s.strip() for s in os.environ.get("DATASET_SPLITS", "train,validation").split(",") if s.strip()]
valid_splits = {"train", "validation", "test"}
splits = [s for s in splits if s in valid_splits]
if not splits:
  splits = ["train", "validation"]

def maybe_build(path_str: str) -> None:
  # We only auto-build files that look like:
  #   coqa_<split>_multiturn.jsonl
  #   quac_<split>_multiturn.jsonl
  # Paths in config are treated as project-root relative (e.g., baseline/one_click/data/...).
  p = (project_root / path_str).resolve() if not os.path.isabs(path_str) else Path(path_str)

  # Deduplicate targets to avoid rebuilding the same file twice.
  global_seen = globals().setdefault("_ONE_CLICK_DATA_PREP_SEEN", set())
  p_key = str(p)
  if p_key in global_seen:
    return
  global_seen.add(p_key)
  force = os.environ.get("FORCE_DATA_PREP", "false").lower() == "true"
  min_lines = int(os.environ.get("MIN_DATASET_LINES", "10"))
  if p.exists() and not force:
    # Heuristic: if a file exists but looks like a tiny smoke artifact, rebuild it.
    try:
      with p.open("r", encoding="utf-8") as f:
        for idx, _ in enumerate(f, start=1):
          if idx >= min_lines:
            return
      # fewer than min_lines
      print(f"[one-click][data] {p} has < {min_lines} lines; rebuilding")
    except Exception:
      return

  m = re.match(r"^(coqa|quac)_(train|validation|test)_(multiturn)\.jsonl$", p.name)
  if not m:
    return

  dataset_name, split, _ = m.group(1), m.group(2), m.group(3)
  p.parent.mkdir(parents=True, exist_ok=True)

  cmd = [
    sys.executable,
    str(build_script),
    "--dataset", dataset_name,
    "--split", split,
    "--out", str(p),
    "--max-conversations", os.environ.get("MAX_CONVERSATIONS", "0"),
    "--max-turns-per-conv", os.environ.get("MAX_TURNS_PER_CONV", "0"),
    "--history-turns", os.environ.get("HISTORY_TURNS", "all"),
  ]

  if p.exists() and force:
    try:
      p.unlink()
    except Exception:
      pass

  print(f"[one-click][data] Building {dataset_name} ({split}) -> {p}")
  try:
    subprocess.check_call(cmd)
  except FileNotFoundError:
    raise
  except subprocess.CalledProcessError as e:
    print(
      "[one-click][data] Failed to build dataset. "
      "Make sure Python package 'datasets' is installed in this environment. "
      "If using conda env: run: pip install datasets.",
      file=sys.stderr,
    )
    raise SystemExit(e.returncode)


for ds in datasets:
  if not isinstance(ds, dict):
    continue
  if ds.get("type") != "jsonl":
    continue
  path = ds.get("path")
  if not path:
    continue
  maybe_build(str(path))

# Also build full train/validation (separate files) if CoQA/QuAC is requested,
# even if the config only points to one split.
for dataset_name in sorted(requested_dataset_names):
  for split in splits:
    out_path = baseline_dir / "one_click" / "data" / f"{dataset_name}_{split}_multiturn.jsonl"
    rel = os.path.relpath(out_path, project_root)
    maybe_build(rel)

print("[one-click][data] Dataset prep done")
PY

  DATA_PREP_ONLY=${DATA_PREP_ONLY:-false}
  if [[ "$DATA_PREP_ONLY" == "true" ]]; then
    echo "[one-click] DATA_PREP_ONLY=true; exiting after dataset prep."
    exit 0
  fi
fi

# -----------------------------------------------------------------------------
# Start vLLM servers (optional)
# -----------------------------------------------------------------------------
if [[ "$SKIP_VLLM" != "true" ]]; then
  echo "[one-click] Starting vLLM servers... (logs in $RUN_DIR)"
  "${PY_CMD[@]}" "$ONE_CLICK_DIR/vllm_launch.py" --config "$CONFIG_PATH" --run-dir "$RUN_DIR"

  # Wait for /v1/models for each endpoint
  echo "[one-click] Waiting for vLLM health..."
  "${PY_CMD[@]}" "$ONE_CLICK_DIR/wait_for_vllm.py" --config "$CONFIG_PATH" --run-dir "$RUN_DIR" --timeout-sec 600
else
  echo "[one-click] SKIP_VLLM=true; assuming endpoints already running."
fi

# -----------------------------------------------------------------------------
# Run online experiment
# -----------------------------------------------------------------------------
echo "[one-click] Running online routing experiment..."
"${PY_CMD[@]}" "$ONE_CLICK_DIR/online_experiment.py" --config "$CONFIG_PATH" --run-dir "$RUN_DIR"

echo "[one-click] Done. Artifacts: $RUN_DIR"
