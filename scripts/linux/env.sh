#!/usr/bin/env bash
# Shared paths for Linux CUDA paper runs. Source from other scripts/linux/*.sh
set -euo pipefail

LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LINUX_DIR/../.." && pwd)"

export ROOT
export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/mempatch:$ROOT/src:$ROOT/scripts}"

# Paper protocol: one fixed stratified train/validation split per backbone.
export PROFILE="${PROFILE:-paper}"
export VALIDATION_PARTS="${VALIDATION_PARTS:-5}"
export VALIDATION_PART="${VALIDATION_PART:-0}"
export SEED="${SEED:-42}"
export RUN_ID="${RUN_ID:-full512}"
export TRAIN_ITERS="${TRAIN_ITERS:-512}"
export SAVE_EVERY="${SAVE_EVERY:-128}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"
export SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
export BASELINE_SET="${BASELINE_SET:-main}"
export INCLUDE_LORA="${INCLUDE_LORA:-0}"
export PRED_TAG_PREFIX="${PRED_TAG_PREFIX:-baseline_}"

# Local artifact layout (under repo-root local/, gitignored)
export LOCAL_ROOT="${LOCAL_ROOT:-$ROOT/local}"
export TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$LOCAL_ROOT/train_data/kfold}"
export ADAPTER_ROOT="${ADAPTER_ROOT:-$LOCAL_ROOT/adapters}"
export RESULTS_ROOT="${RESULTS_ROOT:-$LOCAL_ROOT/results}"
export LOG_ROOT="${LOG_ROOT:-$LOCAL_ROOT/logs/kfold}"
export LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-$LOCAL_ROOT/models}"
export HF_CACHE="${HF_HOME:-$LOCAL_ROOT/hf_cache}"
export PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"

# AutoDL: HF mirror + disable xet CDN (hangs at "Fetching N files: 0%").
# Empty HF_ENDPOINT falls back to mirror; set https://huggingface.co to force official.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT=https://hf-mirror.com
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-1}"

export TEST_SFT_DIR="${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}"

# Resolve dataset paths (hf_release optional; MemPatch.bak clone is a common fallback).
resolve_split_dir() {
  local split="${1:?train|test}"
  local dir
  for dir in \
    "$LOCAL_ROOT/data/mempatch/$split" \
    "$ROOT/local/data/mempatch/$split" \
    "$ROOT/hf_release/mempatch/$split" \
    "$ROOT/../MemPatch.bak/hf_release/mempatch/$split"; do
    if [[ -f "$dir/scenarios.jsonl" ]]; then
      echo "$dir"
      return 0
    fi
  done
  return 1
}

resolve_test_scenarios() {
  if [[ -n "${TEST_SCENARIOS:-}" && -f "$TEST_SCENARIOS" ]]; then
    return 0
  fi
  local candidate
  for candidate in \
    "$TEST_SFT_DIR/scenarios.jsonl" \
    "$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl" \
    "$ROOT/local/data/mempatch/test/scenarios.jsonl" \
    "$ROOT/hf_release/mempatch/test/scenarios.jsonl" \
    "$ROOT/../MemPatch.bak/hf_release/mempatch/test/scenarios.jsonl"; do
    if [[ -f "$candidate" ]]; then
      export TEST_SCENARIOS="$candidate"
      return 0
    fi
  done
  echo "error: test scenarios.jsonl not found." >&2
  echo "  Generate: python scripts/data/generate_mempatch.py --full --out-dir $LOCAL_ROOT/data/mempatch" >&2
  echo "  Or: cp -a ../MemPatch.bak/hf_release/mempatch $LOCAL_ROOT/data/" >&2
  return 1
}

resolve_train_scenarios() {
  if [[ -n "${TRAIN_SCENARIOS:-}" && -f "$TRAIN_SCENARIOS" ]]; then
    return 0
  fi
  local candidate
  for candidate in \
    "$LOCAL_ROOT/data/mempatch/train/scenarios.jsonl" \
    "$ROOT/local/data/mempatch/train/scenarios.jsonl" \
    "$ROOT/hf_release/mempatch/train/scenarios.jsonl" \
    "$ROOT/../MemPatch.bak/hf_release/mempatch/train/scenarios.jsonl"; do
    if [[ -f "$candidate" ]]; then
      export TRAIN_SCENARIOS="$candidate"
      return 0
    fi
  done
  echo "error: train scenarios.jsonl not found." >&2
  echo "  Generate: python scripts/data/generate_mempatch.py --full --out-dir $LOCAL_ROOT/data/mempatch" >&2
  echo "  Or: cp -a ../MemPatch.bak/hf_release/mempatch $LOCAL_ROOT/data/" >&2
  return 1
}

resolve_test_scenarios

mkdir -p "$TRAIN_DATA_ROOT" "$ADAPTER_ROOT" "$RESULTS_ROOT" "$LOG_ROOT" "$HF_CACHE" "$LOCAL_MODEL_ROOT"

source "$LINUX_DIR/model_registry.sh"
