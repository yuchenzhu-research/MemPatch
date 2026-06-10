#!/usr/bin/env bash
# Remove training/eval artifacts under LOCAL_ROOT; keep prefetched model weights.
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   bash scripts/linux/clean_training_artifacts.sh
#
# Keeps:  $LOCAL_ROOT/models/  (snapshot_download weights)
# Removes: adapters, results, logs, train_data, generated runner script
set -euo pipefail
source "$(dirname "$0")/env.sh"

RUNNER="${LOCAL_ROOT}/run_paper_three.sh"

targets=(
  "$ADAPTER_ROOT"
  "$RESULTS_ROOT"
  "$LOG_ROOT"
  "$TRAIN_DATA_ROOT"
  "$(dirname "$PIPELINE_LOG")"
)

echo "LOCAL_ROOT=$LOCAL_ROOT"
echo "Keeping model weights under: $LOCAL_MODEL_ROOT"

for path in "${targets[@]}"; do
  if [[ -d "$path" ]]; then
    echo "rm -rf $path"
    rm -rf "$path"
  fi
done

if [[ -f "$RUNNER" ]]; then
  echo "rm -f $RUNNER"
  rm -f "$RUNNER"
fi

mkdir -p "$ADAPTER_ROOT" "$RESULTS_ROOT" "$LOG_ROOT" "$TRAIN_DATA_ROOT" "$(dirname "$PIPELINE_LOG")"

echo "Done. Models preserved:"
du -sh "$LOCAL_MODEL_ROOT"/* 2>/dev/null || echo "  (no models dir yet)"
