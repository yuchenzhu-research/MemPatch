#!/usr/bin/env bash
# Subset eval: 8 baselines + Path A LoRA (+1). Optional Path B base.
#
#   SLUG=mistral_nemo_12b EVAL_LIMIT=15 bash scripts/linux/run_eval_subset.sh
#   SLUG=mistral_nemo_12b EVAL_LIMIT=25 SKIP_BASE=0 bash scripts/linux/run_eval_subset.sh
#
# Env:
#   EVAL_LIMIT        number of test cases (default 25)
#   EVAL_PREFIX       default test${EVAL_LIMIT}
#   SKIP_BASE         default 1
#   OUTPUT_ROOT       default $LOCAL_ROOT/results_eval_${EVAL_PREFIX}
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_selection.sh"

SLUG="${SLUG:?set SLUG}"
EVAL_LIMIT="${EVAL_LIMIT:-25}"
EVAL_PREFIX="${EVAL_PREFIX:-test${EVAL_LIMIT}}"
SKIP_BASE="${SKIP_BASE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_ROOT/results_eval_${EVAL_PREFIX}}"

export SLUG EVAL_PREFIX EVAL_LIMIT
export SMOKE_RESULT_DIR="$OUTPUT_ROOT/$SLUG"
mkdir -p "$SMOKE_RESULT_DIR" "$LOCAL_ROOT/logs"

log() { echo "[$(date '+%F %T')] [$SLUG] $*"; }

log "output -> $SMOKE_RESULT_DIR"
log "cases: first $EVAL_LIMIT of test500"
log "skip base: $SKIP_BASE"

"$PYTHON" - <<'PY' || { log "error: CUDA required"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"CUDA: {torch.cuda.get_device_name(0)}", file=sys.stderr)
PY

if [[ -n "${BEST_CHECKPOINT:-}" ]]; then
  log "checkpoint: $BEST_CHECKPOINT"
else
  BEST_CHECKPOINT="$(ensure_selection "$SLUG")" || exit 1
  export BEST_CHECKPOINT
  log "checkpoint: $BEST_CHECKPOINT"
fi

if [[ "$SKIP_BASE" != "1" ]]; then
  log "Path B base"
  bash "$LINUX_DIR/06_eval_test.sh" --variant base
fi

log "Path A lora_best (DPA + paired no-DPA)"
PATH_A_STRICT_SMOKE="${PATH_A_STRICT_SMOKE:-1}" \
  bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora

log "8 baselines"
BASELINE_SET=structured_direct,full_context,vanilla_rag,bm25_rag,time_aware_rag,summary_memory,mem0,a_mem \
  PRED_TAG_PREFIX="${EVAL_PREFIX}_baseline_" \
  RESUME=0 INCLUDE_LORA=0 \
  bash "$LINUX_DIR/run_baseline_matrix.sh"

date -Iseconds >"$SMOKE_RESULT_DIR/${EVAL_PREFIX}_8plus1.done"
log "done -> $SMOKE_RESULT_DIR/${EVAL_PREFIX}_8plus1.done"
