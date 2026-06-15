#!/usr/bin/env bash
# Frozen test500 campaign for the two unfinished server models. No training/LoRA.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_gpu.sh"

export RUN_ID="${RUN_ID:-frozen_test500}"
export RESULTS_ROOT="${RESULTS_ROOT:-$LOCAL_ROOT/results/final}"
export TEST_SFT_DIR="${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500_frozen}"
if [[ -z "${TEST_SCENARIOS:-}" ]]; then
  if [[ -f "$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl" ]]; then
    export TEST_SCENARIOS="$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl"
  else
    export TEST_SCENARIOS="$ROOT/supplement/data/test/scenarios.jsonl"
  fi
fi
[[ -f "$TEST_SCENARIOS" ]] || { echo "error: missing test500: $TEST_SCENARIOS" >&2; exit 1; }

for slug in gemma3_12b phi4; do
  echo "===== frozen 5+1 start: $slug ====="
  SLUG="$slug" bash "$LINUX_DIR/prefetch_model.sh"
  BASELINE_SET=main INCLUDE_LORA=0 RESUME=0 SLUG="$slug" \
    bash "$LINUX_DIR/run_baseline_matrix.sh"
  EVAL_PREFIX=test500_mempatch_zero_shot SLUG="$slug" \
    bash "$LINUX_DIR/07_eval_path_a.sh" --variant base
  release_gpu
  echo "===== frozen 5+1 done: $slug ====="
done
