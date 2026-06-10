#!/usr/bin/env bash
# Pick best fold (lowest valid loss) then best checkpoint on that fold.
# Writes local/results/{slug}/kfold_selection.json and checkpoint_selection.json
#
#   SLUG=gemma3_12b bash scripts/linux/05_pick_best.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
RESULT_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$RESULT_DIR"

BEST_FOLD="$("$PYTHON" "$LINUX_DIR/pick_best_kfold_fold.py" \
  --slug "$SLUG" \
  --adapter-root "$ADAPTER_ROOT" \
  --log-root "$LOG_ROOT" \
  --run-id "$RUN_ID" \
  --k-folds "$KFOLDS" \
  --out "$RESULT_DIR/kfold_selection.json")"

BEST_CHECKPOINT="$("$PYTHON" "$LINUX_DIR/pick_best_checkpoint.py" \
  --adapter-dir "$ADAPTER_ROOT/${SLUG}_pathB_lora/fold${BEST_FOLD}/${RUN_ID}" \
  --log-dir "$LOG_ROOT/${SLUG}_fold${BEST_FOLD}" \
  --out "$RESULT_DIR/checkpoint_selection.json")"

export BEST_FOLD BEST_CHECKPOINT
echo "BEST_FOLD=$BEST_FOLD"
echo "BEST_CHECKPOINT=$BEST_CHECKPOINT"
