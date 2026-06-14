#!/usr/bin/env bash
# One multitask adapter per backbone; one selected checkpoint for both adapted interfaces.
set -euo pipefail
export RUN_ID="${RUN_ID:-formal1024}"
export TRAIN_ITERS=1024
export SAVE_EVERY=128
export SAVE_TOTAL_LIMIT=8
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_gpu.sh"
DRY_RUN="${DRY_RUN:-0}"

run() { printf '+ '; printf '%q ' "$@"; printf '\n'; [[ "$DRY_RUN" == "1" ]] || "$@"; }

for slug in "${FORMAL_SLUGS[@]}"; do
  run env SLUG="$slug" PHASES=prefetch,train,pick bash "$LINUX_DIR/run_model.sh"
  # Both commands resolve the same checkpoint_selection.json for this backbone.
  run env SLUG="$slug" EVAL_PREFIX=test500_final_state_control NO_SCHEMA_PROJECTION=1 \
    bash "$LINUX_DIR/06_eval_test.sh" --variant lora
  run env SLUG="$slug" EVAL_PREFIX=test500_mempatch bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora
  release_gpu
done
