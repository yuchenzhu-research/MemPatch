#!/usr/bin/env bash
# Remove partial or mistaken eval/baseline outputs for one SLUG (keeps train/pick).
#
#   SLUG=mistral_nemo_12b bash scripts/linux/clean_eval_artifacts.sh
#   SLUG=mistral_nemo_12b DRY_RUN=1 bash scripts/linux/clean_eval_artifacts.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
RESULT_DIR="$RESULTS_ROOT/$SLUG"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "nothing to clean: $RESULT_DIR"
  exit 0
fi

patterns=(
  smoke.done
  baselines_full.done
  smoke_*
  test500_*
  baseline_*
)

echo "clean eval artifacts for $SLUG in $RESULT_DIR"
for pat in "${patterns[@]}"; do
  shopt -s nullglob
  for f in "$RESULT_DIR"/$pat; do
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  would rm: $f"
    else
      rm -rf "$f"
      echo "  removed: $f"
    fi
  done
  shopt -u nullglob
done

echo "kept: checkpoint_selection.json (selection result)"
echo "kept: adapters under $ADAPTER_ROOT/${SLUG}_pathB_lora/"
