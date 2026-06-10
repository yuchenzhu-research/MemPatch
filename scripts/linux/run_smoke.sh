#!/usr/bin/env bash
# Smoke-test MemPatch (base + LoRA) and all paper baselines with 1 case each.
#
#   SLUG=mistral_nemo_12b bash scripts/linux/run_smoke.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

SLUG="${SLUG:?set SLUG}"
SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
RESULT_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$RESULT_DIR"

rm -f "$RESULT_DIR/smoke.done"

log() { echo "[$(date '+%F %T')] [$SLUG] smoke: $*"; }

log "MemPatch without LoRA (1 case)"
EVAL_PREFIX=smoke_test500 EVAL_LIMIT="$SMOKE_LIMIT" SLUG="$SLUG" \
  bash "$LINUX_DIR/06_eval_test.sh" --variant base

log "MemPatch with LoRA (1 case)"
EVAL_PREFIX=smoke_test500 EVAL_LIMIT="$SMOKE_LIMIT" SLUG="$SLUG" \
  bash "$LINUX_DIR/06_eval_test.sh" --variant lora

log "8 paper baselines (1 case each)"
BASELINE_SET="${BASELINE_SET:-main}" \
  INCLUDE_LORA=0 \
  EVAL_LIMIT="$SMOKE_LIMIT" \
  PRED_TAG_PREFIX=smoke_baseline_ \
  RESUME=0 \
  SLUG="$SLUG" \
  bash "$LINUX_DIR/run_baseline_matrix.sh"

date -Iseconds >"$RESULT_DIR/smoke.done"
log "smoke passed -> $RESULT_DIR/smoke.done"
