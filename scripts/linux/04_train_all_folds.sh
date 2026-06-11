#!/usr/bin/env bash
# Legacy utility for explicit multi-fold experiments; not used by run_model.sh.
#
#   SLUG=gemma3_12b bash scripts/linux/04_train_all_folds.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
EXPERIMENT_FOLDS="${EXPERIMENT_FOLDS:-5}"

for fold in $(seq 0 $((EXPERIMENT_FOLDS - 1))); do
  echo "======== fold $fold / $((EXPERIMENT_FOLDS - 1)) ========"
  VALIDATION_PARTS="$EXPERIMENT_FOLDS" KFOLD_FOLD="$fold" bash "$LINUX_DIR/02_prepare_kfold.sh"
  KFOLD_FOLD="$fold" bash "$LINUX_DIR/03_train_fold.sh"
done

echo "All folds done for $SLUG"
