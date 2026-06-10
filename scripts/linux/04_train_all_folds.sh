#!/usr/bin/env bash
# Train folds 0..4 sequentially.
#
#   SLUG=gemma3_12b bash scripts/linux/04_train_all_folds.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"

for fold in $(seq 0 $((KFOLDS - 1))); do
  echo "======== fold $fold / $((KFOLDS - 1)) ========"
  KFOLD_FOLD="$fold" bash "$LINUX_DIR/02_prepare_kfold.sh"
  KFOLD_FOLD="$fold" bash "$LINUX_DIR/03_train_fold.sh"
done

echo "All folds done for $SLUG"
