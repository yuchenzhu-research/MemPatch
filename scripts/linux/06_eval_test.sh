#!/usr/bin/env bash
# Path B direct-response test500: base model + selected LoRA checkpoint, no DPA.
#
#   SLUG=gemma3_12b bash scripts/linux/06_eval_test.sh
#   SLUG=mistral_nemo_12b bash scripts/linux/06_eval_test.sh --variant base
# Optional: BEST_FOLD=0 BEST_CHECKPOINT=... (from 05_pick_best.sh)
# Env: EVAL_PREFIX=test500  EVAL_LIMIT=1  (smoke uses smoke_test500)
set -euo pipefail
source "$(dirname "$0")/env.sh"

VARIANT_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT_FILTER="${2:?base or lora}"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

SLUG="${SLUG:?set SLUG}"
EVAL_PREFIX="${EVAL_PREFIX:-test500}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
RESULT_DIR="${SMOKE_RESULT_DIR:-$RESULTS_ROOT/$SLUG}"
mkdir -p "$RESULT_DIR"

if [[ ! -f "$TEST_SFT_DIR/sft.jsonl" ]]; then
  echo "Building test SFT bundle -> $TEST_SFT_DIR (from $TEST_SCENARIOS)"
  "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
    --scenarios "$TEST_SCENARIOS" \
    --out-dir "$TEST_SFT_DIR"
fi

# Gold labels for scoring always come from the bundled test split when present.
EVAL_SCENARIOS="$TEST_SCENARIOS"
if [[ -f "$TEST_SFT_DIR/scenarios.jsonl" ]]; then
  EVAL_SCENARIOS="$TEST_SFT_DIR/scenarios.jsonl"
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
  local pred="$RESULT_DIR/${EVAL_PREFIX}_${variant}_predictions.jsonl"
  local metrics="$RESULT_DIR/${EVAL_PREFIX}_${variant}_metrics.json"
  local extra=("$@")

  EVAL_ARGS=(
    --data "$TEST_SFT_DIR/sft.jsonl"
    --eval-data "$EVAL_SCENARIOS"
    --model-id "$HF_MODEL"
    --out-predictions "$pred"
    --out-metrics "$metrics"
    --split-tag "${EVAL_PREFIX}_${variant}"
    --model-tag "$SLUG"
    --variant-tag "$variant"
  )
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    EVAL_ARGS+=(--limit "$EVAL_LIMIT")
  fi
  "$PYTHON" "$LINUX_DIR/run_hf_test_eval.py" "${EVAL_ARGS[@]}" "${extra[@]}"

  "$PYTHON" "$ROOT/scripts/workflows/evaluate_mempatch_predictions.py" \
    --data "$EVAL_SCENARIOS" \
    --predictions "$pred" \
    --no-strict \
    --print-table
}

if [[ -z "$VARIANT_FILTER" || "$VARIANT_FILTER" == "base" ]]; then
  echo "===== base (without adapter) ====="
  run_variant base --no-adapter
fi

if [[ -z "$VARIANT_FILTER" || "$VARIANT_FILTER" == "lora" ]]; then
  echo "===== lora_best (with adapter) ====="
  run_variant lora_best --adapter-path "$ADAPTER_PATH"
fi
