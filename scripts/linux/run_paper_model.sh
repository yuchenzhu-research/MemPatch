#!/usr/bin/env bash
# Full paper pipeline for one model slug on Linux.
#
#   SLUG=gemma3_12b bash scripts/linux/run_paper_model.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"

bash "$LINUX_DIR/01_audit.sh"
bash "$LINUX_DIR/04_train_all_folds.sh"
bash "$LINUX_DIR/05_pick_best.sh"
bash "$LINUX_DIR/06_eval_test.sh"

echo "Done: $SLUG — results in $RESULTS_ROOT/$SLUG"
