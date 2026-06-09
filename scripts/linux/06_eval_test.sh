#!/usr/bin/env bash
# test500: base model + best LoRA checkpoint.
#
#   SLUG=gemma3_12b bash scripts/linux/06_eval_test.sh
# Optional: BEST_FOLD=0 BEST_CHECKPOINT=... (from 05_pick_best.sh)
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
RESULT_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$RESULT_DIR"

if [[ ! -f "$TEST_SFT_DIR/sft.jsonl" ]]; then
  echo "Building test SFT bundle -> $TEST_SFT_DIR"
  "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
    --scenarios "$TEST_SCENARIOS" \
    --out-dir "$TEST_SFT_DIR"
fi

# Resolve best checkpoint from JSON (subprocess export from 05_pick_best.sh is unreliable).
source "$LINUX_DIR/lib_selection.sh"
if [[ -z "${BEST_CHECKPOINT:-}" ]]; then
  BEST_CHECKPOINT="$(ensure_selection "$SLUG")"
fi
ADAPTER_PATH="${BEST_CHECKPOINT:?missing checkpoint; run 05_pick_best.sh first}"

run_variant() {
  local variant="$1"
  shift
  local pred="$RESULT_DIR/test500_${variant}_predictions.jsonl"
  local metrics="$RESULT_DIR/test500_${variant}_metrics.json"
  local extra=("$@")

  EVAL_ARGS=(
    --data "$TEST_SFT_DIR/sft.jsonl"
    --eval-data "$TEST_SCENARIOS"
    --model-id "$HF_MODEL"
    --out-predictions "$pred"
    --out-metrics "$metrics"
    --split-tag "test500_${variant}"
    --model-tag "$SLUG"
    --variant-tag "$variant"
  )
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    EVAL_ARGS+=(--limit "$EVAL_LIMIT")
  fi
  "$PYTHON" "$LINUX_DIR/run_hf_test_eval.py" "${EVAL_ARGS[@]}" "${extra[@]}"

  "$PYTHON" "$ROOT/scripts/workflows/evaluate_mempatch_predictions.py" \
    --data "$TEST_SCENARIOS" \
    --predictions "$pred" \
    --no-strict \
    --print-table
}

echo "===== base (without adapter) ====="
run_variant base --no-adapter

echo "===== lora_best (with adapter) ====="
run_variant lora_best --adapter-path "$ADAPTER_PATH"
