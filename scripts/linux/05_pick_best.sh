#!/usr/bin/env bash
# Pick the lowest-validation-loss checkpoint from the single fixed split.
# Writes local/results/{slug}/checkpoint_selection.json
#
#   SLUG=gemma3_12b bash scripts/linux/05_pick_best.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
RESULT_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$RESULT_DIR"

BEST_FOLD="$VALIDATION_PART"

BEST_CHECKPOINT="$("$PYTHON" "$LINUX_DIR/pick_best_checkpoint.py" \
  --adapter-dir "$ADAPTER_ROOT/${SLUG}_pathB_lora/fold${BEST_FOLD}/${RUN_ID}" \
  --log-dir "$LOG_ROOT/${SLUG}_fold${BEST_FOLD}/${RUN_ID}" \
  --out "$RESULT_DIR/checkpoint_selection.json")"

export BEST_FOLD BEST_CHECKPOINT
echo "BEST_FOLD=$BEST_FOLD"
echo "BEST_CHECKPOINT=$BEST_CHECKPOINT"
